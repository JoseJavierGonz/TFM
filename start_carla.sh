#!/bin/bash
./CARLA_0.9.14/CarlaUE4.sh \
	-quality-level=Low \
	-windowed \
        -ResX=500 -ResY=300 \
	"$@"
