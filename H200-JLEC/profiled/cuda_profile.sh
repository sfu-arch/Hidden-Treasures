#!/bin/sh
# Sets CUDA_VISIBLE_DEVICES based on the current login user.
# Source this from ~/.profile or /etc/profile.d/cuda_assign.sh

GPU_PROFALI="GPU-4b602a03-2530-92db-6933-9832f4f3cc2a"
GPU_PROFRICHARD="GPU-b92a588c-2f85-7851-6552-9137c5be7122"
GPU_ASHRIRAM="GPU-cee43790-22df-153b-e779-8ab0bc1cad31"
GPU_PROFIVAN="GPU-521fdde5-3817-074e-5799-3a16586586a9"

USERS_PROFALI="profali"
USERS_PROFRICHARD="profrichard"
USERS_ASHRIRAM="szm-local"
USERS_PROFIVAN="profivan"

_in_list() { echo " $2 " | grep -qF " $1 "; }
_me="$(id -un)"

if   _in_list "$_me" "$USERS_PROFALI";     then export CUDA_VISIBLE_DEVICES="$GPU_PROFALI"
elif _in_list "$_me" "$USERS_PROFRICHARD"; then export CUDA_VISIBLE_DEVICES="$GPU_PROFRICHARD"
elif _in_list "$_me" "$USERS_ASHRIRAM";    then export CUDA_VISIBLE_DEVICES="$GPU_ASHRIRAM"
elif _in_list "$_me" "$USERS_PROFIVAN";    then export CUDA_VISIBLE_DEVICES="$GPU_PROFIVAN"
fi
