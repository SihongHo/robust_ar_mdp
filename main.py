from env.inventory import *
from env.robot import *
from env.garnet import *
from alg.utility import *
from alg.robust_alg import *
from alg.non_robust_alg import *
import argparse
import datetime
import json
import platform
import warnings
import sys


def main():
    parser = argparse.ArgumentParser(description="Solve problems for different environments.")
    parser.add_argument('--env', choices=['garnet', 'inventory', 'robot'], default='garnet', help='The problem to solve.')
    parser.add_argument('--alg',choices=['robust-base', 'robust-our', 'non-robust'], default='robust-our')
    parser.add_argument('--save_path', type=str, default='./', help='save final results, e.g., exp1/')
    parser.add_argument('--warnings_stop', action='store_true', help='Stop execution on warnings')


    parser.add_argument('--training_steps', type=int, default=50, help='Training steps (default: 50).')
    parser.add_argument('--S_n', type=int, default=3, help='Number of states (default: 3).')
    parser.add_argument('--A_n', type=int, default=2, help='Number of actions (default: 2).')
    parser.add_argument('--uncertainty', type=float, default=0.1, help='Uncertainty factor (default: 0.1).')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate (default: 0.1).')
    parser.add_argument('--max_iterations', type=int, default=1000, help='Maximum number of iterations (default: 1000).')

    args = parser.parse_args()
    # Save the command line arguments and the current time
    run_info = {
        'args': vars(args),
        'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "system": platform.system(),
        "node_name": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
         }
    with open(os.path.join(args.save_path, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=4)

    if args.env == 'garnet':
        env = Garnet(S_n=args.S_n, A_n=args.A_n)
    elif args.env == 'inventory':
        env = Inventory()
    elif args.env == 'robot':
        env = Robot()
    r, P, S_n, A_n = env.get_rewards(), env.get_transition_probabilities(), env.S_n, env.A_n
    print(r)
    print(P)
    pi = generate_random_policy(S_n, A_n)

    training_steps = args.training_steps
    uncertainty = args.uncertainty
    alpha = args.alpha
    save_path = args.save_path

    def warning_stop_run():
        warnings.filterwarnings('error')
        try:
            if args.alg == 'robust-base':
                final_policy, policy_history, V_history = train_robust_base(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
            elif args.alg == 'robust-our':
                final_policy, policy_history, V_history = train_robust(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
            elif args.alg == 'non-robust':
                final_policy, policy_history, V_history = train_non_robust(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
        except Warning as w:
            print(f"Warning occurred: {w}")
            print('finish_time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit("Exiting due to a warning")
    def run():
        if args.alg == 'robust-base':
            final_policy, policy_history, V_history = train_robust_base(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
        elif args.alg == 'robust-our':
            final_policy, policy_history, V_history = train_robust(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
        elif args.alg == 'non-robust':
            final_policy, policy_history, V_history = train_non_robust(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path)
        print('finish_time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.exit("finished")
    
    if args.warnings_stop:
        warning_stop_run()
    else:
        run()

if __name__ == "__main__":
    main()