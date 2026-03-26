#!/bin/bash
RenderOffScreen=$@
if [[ "$RenderOffScreen" == "True" ]]; then
	./CARLA_0.9.14/CarlaUE4.sh \
		-quality-level=Low \
		-windowed \
		-RenderOffScreen \
        	-ResX=600 -ResY=400 

else
	./CARLA_0.9.14/CarlaUE4.sh \
                -quality-level=Low \
                -windowed \
                -ResX=600 -ResY=400
fi
