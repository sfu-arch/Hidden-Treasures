# Dependencies

Rust, LLVM

```
curl https://sh.rustup.rs -sSf | sh
sudo apt-get -y install clang curl llvm-6.0-dev
sudo ln -s /usr/bin/llvm-config-6.0 /usr/bin/llvm-config
```

**If it complains about llvm error check the version of llvm-sys in cargo.toml**.

llvm-sys verions is system llvm version \* 10. If LLVM version is 6.0.0 then
llvm-sys version is 60.0.0
