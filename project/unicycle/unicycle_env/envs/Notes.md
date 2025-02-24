# Notes:

- The generated track is random every episode. 
- Some indicators are shown at the bottom of the window along with the
  state RGB buffer. From left to right: true speed, four ABS sensors,
  steering wheel position, and gyroscope.
- 

## Action Space
If continuous there are 3 actions:
- 0: steering, -1 is full left, +1 is full right
- 1: gas
- 2: braking

If discrete there are 5 actions:
- 0: do nothing
- 1: steer left
- 2: steer right
- 3: gas
- 4: brake

## Observation Space

A top-down 96x96 RGB image of the car and race track.

## Rewards
The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

## Episode Termination
The episode finishes when all the tiles are visited. The car can also go outside the playfield -
that is, far off the track, in which case it will receive -100 reward and die.


## Arguments

- `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
 the agent before a lap is considered complete.
- `domain_randomize=False` enables the domain randomized variant of the environment.
 In this scenario, the background and track colours are different on every reset.
- `continuous=True` converts the environment to use discrete action space.
 The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)





