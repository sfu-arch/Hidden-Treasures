#!/bin/bash
for cpu in /dev/cpu/[0-95]*; do
    wrmsr -p "${cpu##*/}" 0x1a4 15
    rdmsr -p "${cpu##*/}" 0x1a4
done