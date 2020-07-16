## Overview

The purpose of this repository is to build a compiler for a simple toy language. 
Goals of this repository 
- Teach how to parse language using PEG (parsing expression grammars)
- Teach how to generate llvm bit code from language
- Demonstrate how we can avoid register management and phis and work with only allocas, but rely on llvm's optimizer. 

# Dependencies

Rust, LLVM. We expect to find llvm-config in the path.

```bash
$ curl https://sh.rustup.rs -sSf | sh
$ sudo apt-get -y install clang curl llvm-9.0-dev
$ sudo ln -s /usr/bin/llvm-config-9.0 /usr/bin/llvm-config
```

**If it complains about llvm error check the version of llvm-sys in cargo.toml**.

llvm-sys verions is system llvm version \* 10. If LLVM version is 6.0.0 then
llvm-sys version is 60.0.0


## Building compiler

```bash
$ cargo build
$ cargo run in.ex
```

