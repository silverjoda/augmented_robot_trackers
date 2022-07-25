#!/bin/bash

IGN_TRANSPORT_TOPIC_STATISTICS=1 IGN_PARTITION=$(hostname):developer ign service -s /world/example/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 3000 --req 'name: "X1", position: {x: 0, y: 0, z: 0.66}, orientation: {x:0, y:-0.676, z:0, w:0.737}'

#ign service -s /world/example/control --reqtype ignition.msgs.WorldControl --reptype ignition.msgs.Boolean --timeout 5000 --req 'reset: {all: true}'
