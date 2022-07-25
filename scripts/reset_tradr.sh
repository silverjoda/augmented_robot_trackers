#!/bin/bash


ign service -s /world/example/control --reqtype ignition.msgs.WorldControl --reptype ignition.msgs.Boolean --timeout 1000 --req 'reset: {model_only: true}'

ign service -s /world/example/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 2000 --req 'name: "X1", position: {x: -10, y: 0, z: 0.3}, orientation: {x:0, y:0, z:0, w:1}'

