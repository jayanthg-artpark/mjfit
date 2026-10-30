# Opentrons Flex slot

A Franka Panda places a Corning 96-well plate into one Opentrons Flex deck slot. This is
the starting scene for training a contact-based insertion policy instead of grid search.

```bash
git submodule update --init third_party/lab-sentry   # or clone with --recursive
pip install -e . robot_descriptions
python examples/flex_slot/flex_slot.py --view                         # watch one episode
python examples/flex_slot/flex_slot.py --episodes 20 --chamfer 2      # success rates
```

- **Slot.** The corner locators are read from lab-sentry's Flex model
  (`device-designer/models/opentrons_flex/meshes/deck_insert.obj`), a git submodule at
  `third_party/lab-sentry`; `--lab-sentry` points at another checkout. lab-sentry is
  Simon Coelho's and is linked, not copied. Every rib, wall and corner pad is a box or convex prism, so each is one
  convex geom. `--pocket opentrons` (default) moves the corners in to the 128 x 86 mm slot
  of Opentrons' deck definition; `--pocket labsentry` keeps the 140 x 93 mm drawn pocket.
  `--chamfer` bevels the rib tops. The real clips are spring-loaded and their dimensions
  are unmeasured; here they are rigid.
- **Gripper.** The stock Franka Hand opens to 80 mm, narrower than the plate's 85.5 mm, so
  both fingers sit 6 mm further out, like printed lab fingers. They pinch the skirt.
- **Control.** SERL-style Cartesian impedance with gravity compensation. `--xy-stiffness`
  and `--yaw-stiffness` set the compliance while descending.
- **Strategies.** The robot believes the slot is off by up to `--error` mm and `--yaw`
  degrees, then drops straight down or grid-searches around its belief.

## First results

128 x 86 mm pocket, up to 2 mm and 0.3 degrees of error, 20 episodes each. Stiff is
1500 N/m and 40 N m/rad, soft is 200 N/m and 2 N m/rad.

| Chamfer | Gains | Straight drop | Grid search |
| --- | --- | --- | --- |
| none | stiff or soft | 0/20 | 0/20 |
| 1 mm | stiff | 3/20 | 5/20 |
| 1 mm | soft | 0/20 | 4/20 |
| 2 mm | stiff | 5/20 | 7/20 |
| 2 mm | soft | 1/20 | 9/20 |

The 140 x 93 mm pocket seats 20/20 with a straight drop. A rigid pocket with 0.12 mm per
side jams, so these numbers say more about the rigid clips than about the real deck.

## Next

1. Spring-loaded clips and measured slot dimensions.
2. A training environment: Cartesian pose deltas into the impedance controller,
   proprioceptive observations, randomized slot pose, grasp offset, clip stiffness and
   friction; PPO on MuJoCo Warp (needs an NVIDIA GPU).
