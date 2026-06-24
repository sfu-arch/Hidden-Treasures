#!/bin/bash
case "$SLURMD_NODENAME" in
  profali)      echo "export CUDA_VISIBLE_DEVICES=GPU-4b602a03-2530-92db-6933-9832f4f3cc2a" ;;
  profrichard)  echo "export CUDA_VISIBLE_DEVICES=GPU-b92a588c-2f85-7851-6552-9137c5be7122" ;;
  profashriram) echo "export CUDA_VISIBLE_DEVICES=GPU-cee43790-22df-153b-e779-8ab0bc1cad31" ;;
  profivan)     echo "export CUDA_VISIBLE_DEVICES=GPU-521fdde5-3817-074e-5799-3a16586586a9" ;;
esac
