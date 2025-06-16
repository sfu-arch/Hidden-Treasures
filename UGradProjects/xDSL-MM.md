# 🧪 Project Spec: Adding RISC-V AMM Matrix Instructions to xDSL

## 🔍 Overview

In this project, you'll extend the [**xDSL**](https://github.com/xdslproject/xdsl) compiler infrastructure to **support RISC-V AMM matrix multiply instructions**. You'll define new intermediate representations (IRs), implement instruction parsing and verification, and build a simple **software emulator** that mimics the execution of matrix instructions like `mmul`, `macc`, `loadm`, and `storem`.



> 🎥 **Highly Recommended**:
> Watch this MLIR-based talk on mapping `linalg.matmul` to matrix hardware instructions:
> [Commercial RISC-V matrix multiplication](https://www.youtube.com/watch?v=vQ7P5jNy4qE)
> 📘 **Getting Started Tutorial**:
> [xDSL Training Intro](https://github.com/xdslproject/training-intro)

---

## 🔧 What is xDSL?

**xDSL** (eXtensible Domain-Specific Language) is a Python-based compiler infrastructure that mimics [MLIR](https://mlir.llvm.org/), a modern framework for building reusable and composable compiler components.

* xDSL provides a set of **dialects** — namespaces that define operations like arithmetic, control flow, or memory.
* Each operation is modeled as a Python class with input **operands**, output **results**, and optional **attributes**.
* Dialects can be used to represent hardware-specific instructions, which you will do for RISC-V AMM.

---

## 🧩 IRs, Dialects, and ISAs: Similarities and Differences

| Concept     | Description                                                              | Example                    |
| ----------- | ------------------------------------------------------------------------ | -------------------------- |
| **IR**      | A compiler's internal, structured representation of a program            | `%a = addi %x, %y : i32`   |
| **Dialect** | A collection of operations/types that define a specific domain in the IR | `arith`, `memref`, `riscv` |
| **ISA**     | A specification of hardware instructions executed by a processor         | `addi x1, x2, 5` (RISC-V)  |

* **IR**: Abstract representation — architecture-agnostic and often typed.
* **Dialect**: Namespaces within the IR for grouping related operations (e.g., math or hardware-specific).
* **ISA**: The real, binary instruction set a processor decodes and runs.

This project builds a **RISC-V AMM dialect** in the IR, which lets us model matrix instructions before they are lowered to actual RISC-V machine code.

---

## 🤔 What is an IR?

An **Intermediate Representation (IR)** is how compilers represent programs internally. It’s a structured way of describing computation that sits between high-level source code (like Python or C) and low-level machine instructions.

* IRs allow compilers to perform **optimizations** and **transformations**.
* In xDSL, IRs look like typed assembly, e.g.:

  ```mlir
  %c = const 0 : i32
  %a = loadm %ptr : memref<4x4xf32>
  %b = loadm %ptr2 : memref<4x4xf32>
  %r = mmul %a, %b : matrix<4x4xf32>
  ```

---

## 🏠 RISC-V AMM Instructions

The **RISC-V Architected Matrix Multiply (AMM)** extension adds fixed-shape matrix ops to accelerate AI workloads:

| Instruction | Description                         |
| ----------- | ----------------------------------- |
| `loadm`     | Load matrix from memory             |
| `storem`    | Store matrix to memory              |
| `mmul`      | Matrix multiplication (`C = A × B`) |
| `macc`      | Matrix accumulate (`C += A × B`)    |

---

## 📚 Learning Objectives and Skills Development

### 🎯 Core Technical Learning Objectives

By the end of this project, you will be able to:

**Compiler and Systems Knowledge:**
* Explain how compilers use intermediate representations (IRs) to optimize and transform code
* Understand the role of dialects in modular compiler design
* Implement new language constructs in a compiler framework
* Design and implement custom type systems for domain-specific operations

**Software Engineering Skills:**
* Navigate and contribute to a large, well-structured codebase (10,000+ lines)
* Write comprehensive unit tests using pytest framework
* Use version control (Git) effectively for feature development
* Debug complex software systems using both print debugging and Python debugger

**Hardware and Architecture Understanding:**
* Understand how matrix operations map to specialized hardware instructions
* Learn about RISC-V instruction set architecture and extensions
* Appreciate the relationship between high-level operations and low-level execution

**Mathematical Computing:**
* Implement efficient matrix operations using NumPy
* Understand matrix multiplication algorithms and optimizations
* Build software emulators that model hardware behavior

### 💼 Professional Skills Developed

**Problem Solving:**
- Breaking down complex problems into manageable components
- Debugging systematic issues in layered software systems
- Research and understand technical specifications independently

**Communication:**
- Write clear technical documentation with examples
- Explain complex technical concepts to different audiences
- Document design decisions and trade-offs

**Collaboration:**
- Work with established coding standards and conventions
- Contribute to open-source projects following best practices
- Review and understand code written by others

### 🏆 Career-Relevant Outcomes

**For Students Interested in:**

**Systems Programming / Compiler Engineering:**
- Direct experience with compiler infrastructure
- Understanding of how programming languages are implemented
- Knowledge of optimization techniques and IR design

**Computer Architecture / Hardware:**
- Exposure to instruction set design and hardware acceleration
- Understanding of how software interfaces with specialized hardware
- Knowledge of matrix computation architectures (relevant for AI chips)

**Software Engineering / Tech Industry:**
- Large codebase navigation and contribution skills
- Test-driven development practices
- Open-source contribution experience

**Research / Graduate School:**
- Experience with research-grade tools and methodologies
- Understanding of how academic projects translate to practical implementations
- Exposure to cutting-edge compiler and architecture research

### 📈 Skill Progression Throughout Project

**Weeks 1-2 (Foundation):**
- Basic understanding of compilers and IRs
- Python environment management
- Reading and understanding technical documentation

**Weeks 3-4 (Exploration):**
- Code navigation in large projects
- Pattern recognition in software design
- Technical specification analysis

**Weeks 5-7 (Implementation):**
- Object-oriented design and implementation
- Type system design
- Parser and verification logic

**Weeks 8-9 (Integration & Testing):**
- Software emulation and interpretation
- Comprehensive testing strategies
- Performance validation and debugging

**Week 10 (Professional Skills):**
- Technical writing and documentation
- Code review and presentation skills
- Project management and deliverable completion

### 🎓 Assessment and Portfolio Value

**Demonstrable Outcomes for Resume/Portfolio:**
- "Extended xDSL compiler framework to support RISC-V matrix instructions"
- "Implemented software emulator for specialized hardware operations"
- "Contributed 1000+ lines of production-quality Python code with comprehensive tests"
- "Designed custom type system and verification logic for domain-specific language"

**Interview Talking Points:**
- How you approached learning a large, unfamiliar codebase
- Debugging strategies for complex software systems
- Trade-offs in language design and implementation
- Experience with test-driven development and code quality

**Graduate School Applications:**
- Demonstrates research capability and independent learning
- Shows ability to work with academic-grade tools and methodologies
- Provides concrete example of systems-level programming experience

---

## 🎯 Project Goals and Deliverables

1. Extend the RISC-V dialect in xDSL to support AMM matrix instructions.
2. Implement matrix types (`matrix<4x4xf32>`), verification, and parsing.
3. Write an **interpreter/emulator** that executes these matrix ops.
4. Demonstrate functionality with IR-level test programs.
5. Document your code and workflow.

---

## 🎨 ASCII Diagram – Where This Fits

```text
+---------------------+
|  High-Level Dialect |  e.g. linalg.matmul
+----------+----------+
           |
           v
+----------+----------+
|  Lowered IR         |  (your work in RISC-V dialect)
|  mmul, macc, loadm  |
+----------+----------+
           |
           v
+----------+----------+
|  Interpreter / HW   |  NumPy emulator OR real hardware (future)
+---------------------+
```

---

## 🔟 Implementation Tasks (Detailed Breakdown)

### 1. **Research and Specification** (Week 4)
   - [ ] Read the RISC-V AMM specification thoroughly
   - [ ] Create a reference table of instruction formats and opcodes
   - [ ] Document expected behavior for each instruction with examples
   - **Deliverable**: AMM instruction reference document

### 2. **Type System Extension** (Week 5)
   - [ ] Study existing type definitions in `xdsl/dialects/builtin.py`
   - [ ] Implement `MatrixType` class with shape and element type parameters
   - [ ] Add parsing methods: `parse_parameters()` and `print_parameters()`
   - [ ] Include verification for valid matrix dimensions (e.g., 4x4, 8x8)
   - **Code Location**: `xdsl/dialects/riscv.py`
   - **Example**: `matrix<4x4xf32>`, `matrix<8x8xi16>`

### 3. **Operation Definitions** (Week 6)
   For each operation, implement the following structure:
   
   **LoadMatrixOp** (`loadm`):
   - [ ] Operands: memory address (`memref` type)
   - [ ] Results: loaded matrix (`MatrixType`)
   - [ ] Attributes: matrix shape, stride information
   
   **StoreMatrixOp** (`storem`):
   - [ ] Operands: matrix data, memory address
   - [ ] Results: none (side effect operation)
   - [ ] Attributes: matrix shape, stride information
   
   **MatrixMultiplyOp** (`mmul`):
   - [ ] Operands: two input matrices
   - [ ] Results: result matrix
   - [ ] Verification: ensure matrix dimensions are compatible (M×K * K×N = M×N)
   
   **MatrixAccumulateOp** (`macc`):
   - [ ] Operands: accumulator matrix, two input matrices
   - [ ] Results: updated accumulator
   - [ ] Verification: all matrices must have compatible shapes

### 4. **Parser and Printer Implementation** (Week 7)
   - [ ] Define textual syntax for each operation following xDSL conventions
   - [ ] Implement `parse()` class methods for custom syntax parsing
   - [ ] Implement `print()` methods for readable IR output
   - [ ] Handle error cases with informative error messages
   - **Example Syntax**: `%result = riscv.mmul %a, %b : matrix<4x4xf32>`

### 5. **Verification Logic** (Week 7)
   - [ ] Check operand count and types match operation requirements
   - [ ] Verify matrix shape compatibility for multiplication
   - [ ] Ensure element types are compatible (f32, i16, etc.)
   - [ ] Add helpful error messages for debugging
   - **Implementation**: Override `verify_()` method in each operation class

### 6. **Software Emulator/Interpreter** (Week 8)
   - [ ] Create interpreter framework that can execute IR programs
   - [ ] Map each operation to NumPy equivalent:
     - `loadm` → `np.load()` or memory access simulation
     - `storem` → `np.save()` or memory write simulation  
     - `mmul` → `np.matmul(a, b)`
     - `macc` → `c += np.matmul(a, b)`
   - [ ] Handle memory management and matrix storage
   - [ ] Add execution tracing for debugging

### 7. **Comprehensive Testing** (Week 9)
   **Unit Tests** (for each operation):
   - [ ] Valid operation creation and verification
   - [ ] Invalid operand types (should fail gracefully)
   - [ ] Incompatible matrix shapes (should raise verification errors)
   - [ ] Parsing and printing round-trip tests
   
   **Integration Tests**:
   - [ ] Multi-operation IR programs
   - [ ] Memory load/store sequences
   - [ ] Matrix computation chains
   - **Test Location**: `tests/dialects/test_riscv_amm.py`

### 8. **Example Programs and Documentation** (Week 9-10)
   - [ ] Write sample IR programs demonstrating each operation
   - [ ] Create tutorial showing how to use the new operations
   - [ ] Document the emulator usage and capabilities
   - **Examples**:
     ```mlir
     // Matrix multiplication example
     %a = riscv.loadm %ptr_a : matrix<4x4xf32>
     %b = riscv.loadm %ptr_b : matrix<4x4xf32>  
     %c = riscv.mmul %a, %b : matrix<4x4xf32>
     riscv.storem %c, %ptr_c : matrix<4x4xf32>
     ```

### 9. **Integration with xDSL Infrastructure** (Week 10)
   - [ ] Register new operations in the RISC-V dialect
   - [ ] Ensure operations work with xDSL's analysis and transformation passes
   - [ ] Add operations to the dialect's `__init__.py` exports
   - [ ] Test integration with existing xDSL tools and runners

### 10. **Documentation and Code Quality** (Week 10)
   - [ ] Add comprehensive docstrings with usage examples
   - [ ] Include type hints for all method signatures
   - [ ] Write technical documentation explaining design decisions
   - [ ] Create user guide with tutorials and examples
   - [ ] Ensure code follows xDSL style guidelines

---

## 🗂️ Suggested Dialects to Study in xDSL

### 1. Snitch Dialect

* **Path**: `xdsl/dialects/snitch`
* Models instructions similar to matrix operations and embedded processors.

### 2. RISC-V Dialect

* **Path**: `xdsl/dialects/riscv`
* Provides scalar arithmetic and memory ops to extend for AMM.

### 3. Arith Dialect

* **Path**: `xdsl/dialects/arith`
* Defines simple arithmetic ops like `add`, `mul`, `constant`.

### 4. MemRef Dialect

* **Path**: `xdsl/dialects/memref`
* Models memory accesses, needed for `loadm` and `storem`.

### 5. Linalg Dialect (from MLIR)

* **Reference**: [https://mlir.llvm.org/docs/Dialects/Linalg/](https://mlir.llvm.org/docs/Dialects/Linalg/)
* Useful for understanding `linalg.matmul` to `mmul` lowering.

---

## 🎓 Getting Started for Second-Year Undergraduates

This section provides a **structured roadmap** for second-year computer science undergraduates to build the necessary background and complete this project successfully. The path is divided into phases with clear milestones.

### 📋 Prerequisites Assessment

Before starting, you should be comfortable with:
- [ ] **Python programming** (classes, inheritance, basic OOP)
- [ ] **Basic command line usage** (cd, ls, git commands)
- [ ] **Git fundamentals** (clone, commit, push, pull)
- [ ] **Matrix operations** from linear algebra (multiplication, addition)

**If you need review**: Complete a Python refresher course (like MIT 6.0001 or CS50P) before proceeding.

---

### 🚀 Phase 1: Foundation Building (Weeks 1-2)

#### Week 1: Understanding Compilers and IRs

**Learning Goals:**
- Understand what compilers do at a high level
- Learn what Intermediate Representations (IRs) are and why they exist
- Get familiar with the concept of dialects in compiler design

**Tasks:**

1. **📚 Read and Take Notes** (3-4 hours)
   - Read ["What is a Compiler?"](https://www.tutorialspoint.com/compiler_design/compiler_design_overview.htm)
   - Watch: ["Compilers Explained" by Computerphile](https://www.youtube.com/watch?v=IhC7sdYe-Jg) (10 min)
   - Read the MLIR introduction: [MLIR Overview](https://mlir.llvm.org/) (just the overview section)

2. **🔍 Explore xDSL Basics** (2-3 hours)
   - Clone the xDSL repository: `git clone https://github.com/xdslproject/xdsl.git`
   - Read the main README.md file
   - Browse through `xdsl/dialects/arith.py` to see how simple operations are defined
   - **Deliverable**: Write a 1-page summary of what you learned about IRs and dialects

3. **⚙️ Environment Setup** (1-2 hours)
   - Install Python 3.10+ and pip
   - Set up a virtual environment: `python -m venv xdsl-env`
   - Activate environment: `source xdsl-env/bin/activate` (macOS/Linux)
   - Install xDSL in development mode: `pip install -e .` (from xDSL directory)
   - **Milestone**: Successfully run `python -c "import xdsl; print('xDSL imported successfully!')"`

#### Week 2: Matrix Operations and RISC-V Basics

**Learning Goals:**
- Refresh matrix multiplication concepts
- Understand RISC-V instruction set basics
- Learn about hardware acceleration for matrix operations

**Tasks:**

1. **🧮 Matrix Math Review** (2-3 hours)
   - Review matrix multiplication using Khan Academy's [Matrix Multiplication](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices)
   - Practice with NumPy: Write a Python script that multiplies two 4x4 matrices
   - **Deliverable**: Python script showing manual matrix multiplication vs NumPy

2. **🏗️ RISC-V Introduction** (3-4 hours)
   - Read Chapter 1 of ["RISC-V Reader"](https://riscvbook.com/) (available free online)
   - Watch: ["RISC-V in 5 minutes" overview video
   - Browse the [RISC-V Instruction Set Manual](https://riscv.org/technical/specifications/) (just skim for now)
   - **Deliverable**: Write 2-3 paragraphs explaining what RISC-V is and why it's important

3. **🎯 Matrix Hardware Acceleration** (2 hours)
   - Read about why matrix operations need hardware acceleration
   - Watch the recommended video: [Mapping High-Level Linalg to Matrix Instructions](https://www.youtube.com/watch?v=vQ7P5jNy4qE)
   - **Milestone**: Can explain in your own words why GPUs and specialized hardware are needed for AI workloads

---

### 🛠️ Phase 2: Hands-On xDSL Exploration (Weeks 3-4)

#### Week 3: Diving into xDSL Code

**Learning Goals:**
- Navigate the xDSL codebase confidently
- Understand how operations are defined and work
- See how existing dialects are structured

**Tasks:**

1. **🔍 Code Exploration** (4-5 hours)
   - Study `xdsl/dialects/arith.py` line by line
   - Understand the structure: imports, operation definitions, dialect registration
   - Look at how `AddIOp`, `MulIOp` are defined
   - **Exercise**: Write pseudocode for how you would add a `square` operation to the arith dialect

2. **📝 Operation Anatomy** (3-4 hours)
   - Read `xdsl/irdl/operations.py` to understand the base `Operation` class
   - Study how operations have operands, results, and attributes
   - Look at examples in `xdsl/dialects/builtin.py`
   - **Deliverable**: Create a diagram showing the components of an xDSL operation

3. **🧪 Running Tests** (2 hours)
   - Navigate to the xDSL directory
   - Run: `python -m pytest tests/dialects/test_arith.py -v`
   - Understand what the tests are checking
   - **Milestone**: All arith dialect tests pass on your machine

#### Week 4: Understanding RISC-V Dialect

**Learning Goals:**
- Examine the existing RISC-V dialect in xDSL
- Understand how instruction sets map to IR operations
- Prepare for extending the dialect

**Tasks:**

1. **🔎 RISC-V Dialect Analysis** (4-5 hours)
   - Study `xdsl/dialects/riscv.py` thoroughly
   - Identify patterns: how instructions are named, what operands they take
   - Look at `AddOp`, `SubOp`, `LoadOp` examples
   - **Exercise**: Pick 3 operations and explain their operands and semantics

2. **📊 Matrix Operations Research** (3-4 hours)
   - Research RISC-V AMM (Architected Matrix Multiply) specification
   - Find documentation about `loadm`, `storem`, `mmul`, `macc` instructions
   - Understand what operands and attributes these would need
   - **Deliverable**: Table showing each AMM instruction with its inputs, outputs, and behavior

3. **🎨 Design Planning** (2-3 hours)
   - Sketch out how you would add `MatrixType` to xDSL
   - Plan the structure of your four new operations
   - **Milestone**: Written plan for your implementation approach

---

### ⚡ Phase 3: Implementation (Weeks 5-7)

#### Week 5: Matrix Type Implementation

**Learning Goals:**
- Add a new type to the xDSL type system
- Understand type verification and parsing

**Tasks:**

1. **📐 MatrixType Design** (2-3 hours)
   - Study how types are defined in `xdsl/dialects/builtin.py`
   - Look at `IntegerType`, `FloatType` examples
   - Plan your `MatrixType` with shape and element type parameters

2. **💻 MatrixType Implementation** (4-6 hours)
   - Create `MatrixType` class in the RISC-V dialect
   - Implement parsing and printing methods
   - Add verification logic for valid shapes and element types
   - **Milestone**: Can create and print `matrix<4x4xf32>` types

3. **🧪 Testing MatrixType** (2-3 hours)
   - Write unit tests for your MatrixType
   - Test edge cases: invalid shapes, unsupported element types
   - **Deliverable**: Working MatrixType with passing tests

#### Week 6: Matrix Operations

**Learning Goals:**
- Implement the four AMM operations
- Add proper operand and result type handling

**Tasks:**

1. **⚙️ LoadMatrixOp Implementation** (3-4 hours)
   - Define `LoadMatrixOp` class following xDSL patterns
   - Specify operands (memory address) and results (matrix)
   - Implement verification logic
   - **Milestone**: Can parse and verify loadm operations

2. **💾 StoreMatrixOp Implementation** (3-4 hours)
   - Define `StoreMatrixOp` class
   - Handle matrix input and memory address operands
   - **Milestone**: Can parse and verify storem operations

3. **🔢 MatrixMultiplyOp Implementation** (4-5 hours)
   - Define `MatrixMultiplyOp` class
   - Ensure matrix dimensions are compatible for multiplication
   - **Milestone**: Can parse and verify mmul operations

4. **➕ MatrixAccumulateOp Implementation** (3-4 hours)
   - Define `MatrixAccumulateOp` class
   - Handle accumulation logic (C += A × B)
   - **Deliverable**: All four operations implemented and tested

#### Week 7: Parser and Verification

**Learning Goals:**
- Complete the parsing and pretty-printing functionality
- Ensure robust error handling

**Tasks:**

1. **📝 Parser Implementation** (4-5 hours)
   - Add parsing methods for your operations
   - Follow xDSL parsing conventions
   - Handle syntax errors gracefully

2. **✅ Verification Completion** (3-4 hours)
   - Add comprehensive verification for matrix shape compatibility
   - Check operand and result types match expectations
   - **Milestone**: All operations parse, print, and verify correctly

3. **🔍 Integration Testing** (2-3 hours)
   - Test operations together in small IR programs
   - **Deliverable**: Sample IR program using all four operations

---

### 🎯 Phase 4: Emulation and Testing (Weeks 8-9)

#### Week 8: Software Emulator

**Learning Goals:**
- Build an interpreter for your matrix operations
- Use NumPy for actual matrix computations

**Tasks:**

1. **🧮 NumPy Review** (2-3 hours)
   - Review NumPy matrix operations: `np.matmul`, `np.add`
   - Practice loading/storing matrix data
   - **Exercise**: Implement matrix multiplication manually, then with NumPy

2. **⚙️ Emulator Design** (3-4 hours)
   - Study how interpreters work in xDSL
   - Plan how to map IR operations to NumPy calls
   - Design memory model for matrix storage

3. **💻 Emulator Implementation** (4-6 hours)
   - Implement interpreter functions for each operation
   - Handle memory management for matrices
   - **Milestone**: Can execute simple matrix IR programs

#### Week 9: Comprehensive Testing

**Learning Goals:**
- Validate your implementation thoroughly
- Create comprehensive test suite

**Tasks:**

1. **🧪 Unit Test Suite** (4-5 hours)
   - Write tests for each operation individually
   - Test error conditions and edge cases
   - **Deliverable**: Comprehensive test suite with >90% coverage

2. **🎯 Integration Examples** (3-4 hours)
   - Create example IR programs that use multiple operations
   - Test realistic matrix computation scenarios
   - **Milestone**: Can run complex matrix programs end-to-end

3. **📊 Performance Validation** (2-3 hours)
   - Compare your emulator results with direct NumPy
   - Ensure numerical accuracy
   - **Deliverable**: Validation report showing correctness

---

### 📚 Phase 5: Documentation and Presentation (Week 10)

**Learning Goals:**
- Document your work professionally
- Prepare for code review and presentation

**Tasks:**

1. **📖 Code Documentation** (3-4 hours)
   - Add comprehensive docstrings to all classes and methods
   - Include usage examples in documentation
   - **Deliverable**: Well-documented code ready for review

2. **📝 Project Report** (4-5 hours)
   - Write a technical report explaining your implementation
   - Include design decisions, challenges, and solutions
   - Add performance analysis and future work sections
   - **Deliverable**: 5-10 page technical report

3. **🎤 Presentation Preparation** (2-3 hours)
   - Create slides demonstrating your project
   - Prepare live demo of matrix operations
   - **Milestone**: Ready to present your work

---

### 🎯 Weekly Milestones and Checkpoints

| Week | Key Milestone | Deliverable |
|------|---------------|-------------|
| 1 | Environment setup complete | xDSL running locally |
| 2 | Background knowledge acquired | Matrix multiplication script |
| 3 | xDSL codebase understood | Operation anatomy diagram |
| 4 | Extension plan ready | AMM instruction specification |
| 5 | MatrixType working | Passing type tests |
| 6 | All operations implemented | Four AMM operations |
| 7 | Parsing/verification complete | Sample IR program |
| 8 | Emulator functional | Working interpreter |
| 9 | Testing complete | Comprehensive test suite |
| 10 | Project documented | Final report and presentation |

---

### 🆘 Getting Help and Resources

**When You're Stuck:**
1. **Read the error message carefully** - xDSL provides detailed error messages
2. **Check existing examples** - Look at similar operations in other dialects
3. **Use Python debugger** - Add `import pdb; pdb.set_trace()` to debug
4. **Ask specific questions** - Don't just say "it doesn't work"

**Additional Resources:**
- xDSL Documentation: [xdslproject.github.io](https://xdslproject.github.io/)
- MLIR Language Reference: [mlir.llvm.org/docs/LangRef/](https://mlir.llvm.org/docs/LangRef/)
- Python Class Design: [Real Python OOP](https://realpython.com/python3-object-oriented-programming/)
- NumPy Documentation: [numpy.org/doc/](https://numpy.org/doc/)

**Time Management Tips:**
- **Budget 10-15 hours per week** for this project
- **Start early** - compiler projects often take longer than expected
- **Test frequently** - Don't write everything before testing
- **Document as you go** - Don't leave documentation until the end

---

## 🚨 Common Issues and Troubleshooting

### 🔧 Installation and Setup Problems

**Issue: "ModuleNotFoundError: No module named 'xdsl'"**
```bash
# Solution: Make sure you're in the right directory and virtual environment
cd /path/to/xdsl
source xdsl-env/bin/activate  # Activate virtual environment
pip install -e .              # Install in development mode
python -c "import xdsl; print('Success!')"
```

**Issue: "Python version incompatible"**
```bash
# Solution: Check Python version and upgrade if needed
python --version  # Should be 3.10+
# If too old, install newer Python and recreate virtual environment
```

**Issue: "Permission denied" when running commands**
```bash
# Solution: Don't use sudo with pip in virtual environments
# Make sure you're in your virtual environment first
which python  # Should show path to venv, not system Python
```

### 🐛 Common Coding Errors

**Error: "VerificationError: Operation does not match type signature"**
```python
# Problem: Operand types don't match what operation expects
# Check that you're passing the right types to operations

# Wrong:
matrix_val = SSAValue(IntegerType(32))  # This is an integer, not matrix!
mmul_op = MatrixMultiplyOp([matrix_val, matrix_val], [...])

# Right: 
matrix_val = SSAValue(MatrixType([4, 4], f32))  # Correct matrix type
mmul_op = MatrixMultiplyOp([matrix_val, matrix_val], [...])
```

**Error: "AttributeError: 'MatrixType' object has no attribute 'shape'"**
```python
# Problem: You haven't implemented all required methods/attributes
# Make sure your MatrixType has all the required fields

@irdl_attr_definition
class MatrixType(Data[MatrixType]):
    name = "riscv.matrix"
    shape: ParametrizedAttribute = param_def(ArrayAttr)  # Don't forget this!
    element_type: ParametrizedAttribute = param_def(Attribute)
```

**Error: "ParseError: Expected 'x' but found 'y'"**
```python
# Problem: Your parser doesn't match the syntax you're trying to parse
# Check that parse_parameters matches the syntax exactly

# If you want to parse: matrix<4x4xf32>
# Your parser should handle: '4', 'x', '4', 'x', 'f32'
```

### 🧪 Testing Issues

**Issue: "Tests pass individually but fail when run together"**
- **Cause**: Global state or side effects between tests
- **Solution**: Make sure each test is independent, use fresh objects

**Issue: "ImportError in test files"**
- **Cause**: Python can't find your new modules
- **Solution**: Run tests from the xDSL root directory, check `__init__.py` files

**Issue: "Test coverage too low"**
- **Solution**: Add tests for error cases, edge cases, and all public methods

### 💡 Design Decision Guidance

**Question: "Should matrices be fixed-size or variable-size?"**
- **Recommendation**: Start with fixed sizes (4x4, 8x8) to match RISC-V AMM spec
- **Future**: Can extend to variable sizes later

**Question: "What element types should I support?"**
- **Start with**: `f32` (32-bit float) and `i16` (16-bit integer)
- **Reasoning**: These are common in AI workloads and easy to test

**Question: "How detailed should verification be?"**
- **Minimum**: Check operand count, basic type compatibility
- **Recommended**: Also check matrix shape compatibility, element type matching
- **Advanced**: Check memory alignment, stride compatibility

### 📞 When to Ask for Help

**Ask Immediately If:**
- You can't get the basic xDSL installation working after 2+ hours
- You're getting Python import errors that don't make sense
- The existing xDSL tests are failing on your machine

**Try for 1-2 Hours First:**
- Debugging your own code logic errors
- Understanding how existing xDSL operations work
- Writing test cases for your operations

**Research First, Then Ask:**
- How to implement specific language features
- Best practices for compiler design
- Performance optimization questions

### 🎯 Project Scope Management

**Minimum Viable Project (if running short on time):**
- [ ] MatrixType with basic shape and element type
- [ ] MatrixMultiplyOp with verification
- [ ] Simple NumPy-based emulator for mmul
- [ ] Basic tests demonstrating functionality

**Full Project Scope:**
- [ ] All four operations (loadm, storem, mmul, macc)
- [ ] Complete parser and printer support
- [ ] Comprehensive verification logic
- [ ] Full software emulator with memory management
- [ ] Extensive test suite with edge cases

**Extension Ideas (if ahead of schedule):**
- [ ] Support for different matrix shapes (not just 4x4)
- [ ] Multiple element types (int8, int16, f16, f32)
- [ ] Memory layout optimizations (row-major vs column-major)
- [ ] Integration with xDSL optimization passes

### 📋 Pre-Submission Checklist

**Code Quality:**
- [ ] All methods have docstrings with examples
- [ ] Code follows Python naming conventions (snake_case)
- [ ] No hardcoded values (use constants or parameters)
- [ ] Error messages are helpful and specific

**Testing:**
- [ ] All operations can be created and verified
- [ ] Parser/printer round-trip works (parse then print gives same result)
- [ ] Error cases raise appropriate exceptions
- [ ] Integration tests show operations working together

**Documentation:**
- [ ] README explains how to run your code
- [ ] Example IR programs demonstrate all operations
- [ ] Technical report explains design decisions
- [ ] Code comments explain complex logic

**Functionality:**
- [ ] Can parse and verify all matrix operations
- [ ] Emulator correctly computes matrix operations
- [ ] Results match NumPy reference implementations
- [ ] Handles edge cases gracefully

