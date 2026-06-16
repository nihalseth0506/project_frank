import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from environment.pick_place_scripted_env import ScriptedPickPlace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args()

    print("=" * 55)
    print("FRANK — Scripted Pick and Place")
    print("Panda reach policy + scripted grasp/lift/place")
    print("=" * 55)

    controller    = ScriptedPickPlace(render=not args.no_render)
    total_success = 0
    results       = []

    for ep in range(args.episodes):
        print(f"\n{'─' * 40}")
        print(f"Episode {ep+1}/{args.episodes}")

        success = controller.run_episode(max_steps=args.max_steps)
        total_success += int(success)
        results.append(success)

        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")

        if not args.no_render:
            if controller.viewer and not controller.viewer.is_running():
                break

    print(f"\n{'═' * 55}")
    print(f"Success rate: {total_success}/{len(results)} = "
          f"{total_success/max(len(results),1)*100:.1f}%")

    controller.close()


if __name__ == "__main__":
    main()