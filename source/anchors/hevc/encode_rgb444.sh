#!/bin/sh

./hevc/bin/TAppEncoderStatic -c ./hevc/cfg/encoder_lowdelay_P_main.cfg -c ./hevc/cfg/formatRGB.cfg -c ./hevc/cfg/per_sequence/inputrgb444.cfg -q $1 -f $2 -wdt $3 -hgt $4 -i $5 -o $6/$7_qp$1.yuv -b $6/$7_qp$1.bin >>$6/$7_qp$1.log 