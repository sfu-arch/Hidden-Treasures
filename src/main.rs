extern crate peg;

extern crate llvm_sys as llvm;
pub use llvm::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::ptr;

pub enum Expr {
    Literal(String),
    Identifier(String),
    Assign(String, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    IfElse(Box<Expr>, Vec<Expr>, Vec<Expr>),
    WhileLoop(Box<Expr>, Vec<Expr>),
}

peg::parser!( grammar program() for str {
    use super::Expr;
pub rule statements() -> Vec<Expr>
        = s:(statement()*) { s }

    rule statement() -> Expr
        = _ e:expression() _ "\n" { e }

    rule expression() -> Expr
        = if_else()
        / while_loop()
        / assignment()
        / binary_op()

    rule if_else() -> Expr
        = "if" _ e:expression() _ "{" _ "\n"
        then_body:statements() _ "}" _ "else" _ "{" _ "\n"
        else_body:statements() _ "}"
        { Expr::IfElse(Box::new(e), then_body, else_body) }

    rule while_loop() -> Expr
        = "while" _ e:expression() _ "{" _ "\n"
        loop_body:statements() _ "}"
        { Expr::WhileLoop(Box::new(e), loop_body) }

    rule assignment() -> Expr
        = i:identifier() _ "=" _ e:expression() {Expr::Assign(i, Box::new(e))}

    rule binary_op() -> Expr = precedence!{
        a:@ _ "==" _ b:(@) { Expr::Eq(Box::new(a), Box::new(b)) }
        a:@ _ "!=" _ b:(@) { Expr::Ne(Box::new(a), Box::new(b)) }
        a:@ _ "<"  _ b:(@) { Expr::Lt(Box::new(a), Box::new(b)) }
        a:@ _ "<=" _ b:(@) { Expr::Le(Box::new(a), Box::new(b)) }
        a:@ _ ">"  _ b:(@) { Expr::Gt(Box::new(a), Box::new(b)) }
        a:@ _ ">=" _ b:(@) { Expr::Ge(Box::new(a), Box::new(b)) }
        --
        a:@ _ "+" _ b:(@) { Expr::Add(Box::new(a), Box::new(b)) }
        a:@ _ "-" _ b:(@) { Expr::Sub(Box::new(a), Box::new(b)) }
        --
        a:@ _ "*" _ b:(@) { Expr::Mul(Box::new(a), Box::new(b)) }
        a:@ _ "/" _ b:(@) { Expr::Div(Box::new(a), Box::new(b)) }
        --
        i:identifier() { Expr::Identifier(i) }
        l:literal() { l }
    }
    rule identifier() -> String
        = quiet!{ n:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) { n.to_owned() } }
        / expected!("identifier")

    rule literal() -> Expr
        = n:$(['0'..='9']+) { Expr::Literal(n.to_owned()) }
    rule _() =  quiet!{[' ' | '\t']*}
});

use self::program::*;

fn main() {
    let mut input = String::new();
    let mut f = File::open("in.ex").unwrap();
    f.read_to_string(&mut input).unwrap();

    let parsed_input = statements(&input).unwrap();

    unsafe {
        codegen(parsed_input);
    }
}

unsafe fn codegen(input: Vec<Expr>) {
    let context = llvm::core::LLVMContextCreate();
    let module = llvm::core::LLVMModuleCreateWithName(b"example_module\0".as_ptr() as *const _);
    let builder = llvm::core::LLVMCreateBuilderInContext(context);

    // In LLVM, you get your types from functions.
    let int_type = llvm::core::LLVMInt64TypeInContext(context);
    let function_type = llvm::core::LLVMFunctionType(int_type, ptr::null_mut(), 0, 0);

    let function =
        llvm::core::LLVMAddFunction(module, b"main\0".as_ptr() as *const _, function_type);

    let entry_name = CString::new("entry").unwrap();
    let bb = llvm::core::LLVMAppendBasicBlockInContext(context, function, entry_name.as_ptr());
    llvm::core::LLVMPositionBuilderAtEnd(builder, bb);

    let mut names = HashMap::new();
    insert_allocations(context, builder, &mut names, &input);

    let int_type = llvm::core::LLVMInt64TypeInContext(context);
    let zero = llvm::core::LLVMConstInt(int_type, 0, 0);

    let mut return_value = zero; // return value on empty program
    for expr in input {
        return_value = codegen_expr(context, builder, function, &mut names, expr);
    }
    llvm::core::LLVMBuildRet(builder, return_value);

    // Instead of dumping to stdout, let's write out the IR to `out.ll`
    let out_file = CString::new("out.ll").unwrap();
    llvm::core::LLVMPrintModuleToFile(module, out_file.as_ptr(), ptr::null_mut());

    llvm::core::LLVMDisposeBuilder(builder);
    llvm::core::LLVMDisposeModule(module);
    llvm::core::LLVMContextDispose(context);
}

unsafe fn insert_allocations(
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    names: &mut HashMap<String, LLVMValueRef>,
    exprs: &[Expr],
) {
    let mut variable_names = HashSet::new();
    for expr in exprs {
        match *expr {
            Expr::Assign(ref name, _) => {
                variable_names.insert(name);
            }

            _ => {}
        }
    }

    for variable_name in variable_names {
        let int_type = llvm::core::LLVMInt64TypeInContext(context);
        let name = CString::new(variable_name.as_bytes()).unwrap();
        let pointer = llvm::core::LLVMBuildAlloca(builder, int_type, name.as_ptr());

        names.insert(variable_name.to_owned(), pointer);
    }
}
// When you write out instructions in LLVM, you get back `LLVMValueRef`s. You
// can then use these references in other instructions.
unsafe fn codegen_expr(
    context: LLVMContextRef,
    builder: LLVMBuilderRef,
    func: LLVMValueRef,
    names: &mut HashMap<String, LLVMValueRef>,
    expr: Expr,
) -> LLVMValueRef {
    match expr {
        Expr::Literal(int_literal) => {
            let int_type = llvm::core::LLVMInt64TypeInContext(context);
            llvm::core::LLVMConstInt(int_type, int_literal.parse().unwrap(), 0)
        }

        Expr::Eq(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("eqtmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntEQ;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Ne(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("neqtmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntNE;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Lt(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("lttmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntSLT;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Le(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("lttmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntSLE;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Gt(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("lttmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntSGT;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Ge(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);
            let name = CString::new("lttmp").unwrap();
            let op = llvm::LLVMIntPredicate::LLVMIntSGE;
            llvm::core::LLVMBuildICmp(builder, op, lhs, rhs, name.as_ptr())
        }

        Expr::Add(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);

            let name = CString::new("addtmp").unwrap();
            llvm::core::LLVMBuildAdd(builder, lhs, rhs, name.as_ptr())
        }

        Expr::Sub(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);

            let name = CString::new("subtmp").unwrap();
            llvm::core::LLVMBuildSub(builder, lhs, rhs, name.as_ptr())
        }

        Expr::Mul(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);

            let name = CString::new("multmp").unwrap();
            llvm::core::LLVMBuildMul(builder, lhs, rhs, name.as_ptr())
        }

        Expr::Div(lhs, rhs) => {
            let lhs = codegen_expr(context, builder, func, names, *lhs);
            let rhs = codegen_expr(context, builder, func, names, *rhs);

            let name = CString::new("divtmp").unwrap();
            llvm::core::LLVMBuildUDiv(builder, lhs, rhs, name.as_ptr())
        }
        Expr::Assign(name, expr) => {
            let new_value = codegen_expr(context, builder, func, names, *expr);
            let pointer = names.get(&name).unwrap();
            llvm::core::LLVMBuildStore(builder, new_value, *pointer);
            new_value
        }

        Expr::Identifier(name) => {
            let new_reg = CString::new("assigntmp").unwrap();
            let pointer = names.get(&name).unwrap();
            llvm::core::LLVMBuildLoad(builder, *pointer, new_reg.as_ptr())
        }

        Expr::IfElse(condition, then_body, else_body) => {
            let condition_value = codegen_expr(context, builder, func, names, *condition);
            let int_type = llvm::core::LLVMInt64TypeInContext(context);
            let pred_type = llvm::core::LLVMInt1TypeInContext(context);
            let zero = llvm::core::LLVMConstInt(pred_type, 0, 0);

            let name = CString::new("is_nonzero").unwrap();
            let is_nonzero = llvm::core::LLVMBuildICmp(
                builder,
                llvm::LLVMIntPredicate::LLVMIntNE,
                condition_value,
                zero,
                name.as_ptr(),
            );

            let entry_name = CString::new("entry").unwrap();
            let then_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());
            let else_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());
            let merge_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());

            llvm::core::LLVMBuildCondBr(builder, is_nonzero, then_block, else_block);

            llvm::core::LLVMPositionBuilderAtEnd(builder, then_block);
            let mut then_return = zero;
            for expr in then_body {
                then_return = codegen_expr(context, builder, func, names, expr);
            }
            llvm::core::LLVMBuildBr(builder, merge_block);
            let then_block = llvm::core::LLVMGetInsertBlock(builder);

            llvm::core::LLVMPositionBuilderAtEnd(builder, else_block);
            let mut else_return = zero;
            for expr in else_body {
                else_return = codegen_expr(context, builder, func, names, expr);
            }
            llvm::core::LLVMBuildBr(builder, merge_block);
            let else_block = llvm::core::LLVMGetInsertBlock(builder);

            llvm::core::LLVMPositionBuilderAtEnd(builder, merge_block);
            let phi_name = CString::new("iftmp").unwrap();
            let phi = llvm::core::LLVMBuildPhi(builder, int_type, phi_name.as_ptr());

            let mut values = vec![then_return, else_return];
            let mut blocks = vec![then_block, else_block];

            llvm::core::LLVMAddIncoming(phi, values.as_mut_ptr(), blocks.as_mut_ptr(), 2);
            phi
        }
        Expr::WhileLoop(condition, body) => {
            let entry_name = CString::new("loop").unwrap();
            let header_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());
            let then_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());
            let merge_block =
                llvm::core::LLVMAppendBasicBlockInContext(context, func, entry_name.as_ptr());
            llvm::core::LLVMBuildBr(builder, header_block);
            llvm::core::LLVMPositionBuilderAtEnd(builder, header_block);

            let condition_value = codegen_expr(context, builder, func, names, *condition);
            let int_type = llvm::core::LLVMInt64TypeInContext(context);
            let pred_type = llvm::core::LLVMInt1TypeInContext(context);
            let zero = llvm::core::LLVMConstInt(pred_type, 0, 0);

            let name = CString::new("is_nonzero").unwrap();
            let is_nonzero = llvm::core::LLVMBuildICmp(
                builder,
                llvm::LLVMIntPredicate::LLVMIntNE,
                condition_value,
                zero,
                name.as_ptr(),
            );

            llvm::core::LLVMBuildCondBr(builder, is_nonzero, then_block, merge_block);

            llvm::core::LLVMPositionBuilderAtEnd(builder, then_block);
            let mut then_return = zero;
            for expr in body {
                then_return = codegen_expr(context, builder, func, names, expr);
            }
            llvm::core::LLVMBuildBr(builder, header_block);

            llvm::core::LLVMPositionBuilderAtEnd(builder, merge_block);
            let int_type = llvm::core::LLVMInt64TypeInContext(context);
            llvm::core::LLVMConstInt(int_type, 0, 0)
        }
    }
}
