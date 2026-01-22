#!/bin/sh

./hevc/bin/TAppDecoderStatic -b $3/$1_qp$2.bin  -o $4/$1_qp$2_dec.rgb --OutputColourSpaceConvert=GBRtoRGB > $4/$1_qp$2_dec.log
