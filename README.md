

### To test:

Run the following commands in your terminal:

- For algorithm `our robust algorithm`:
  ```bash
  python main.py --alg robust-our --training_steps 1 --max_iterations 1 --save_path exp1/ --env inventory
  ```
  ```bash
  python main.py --alg robust-our --training_steps 1 --max_iterations 1 --save_path exp1/ --env garnet
  ```
  ```bash
  python main.py --alg robust-our --training_steps 1 --max_iterations 1 --save_path exp1/ --env robot
  ```

- For algorithm `robust baseline`:
  ```bash
  python main.py --alg robust-base --training_steps 1 --max_iterations 1 --save_path exp1/ --env inventory
  ```
  ```bash
  python main.py --alg robust-base --training_steps 1 --max_iterations 1 --save_path exp1/ --env garnet
  ```
  ```bash
  python main.py --alg robust-base --training_steps 1 --max_iterations 1 --save_path exp1/ --env robot
  ```

- For algorithm `non-robust baseline`:
  ```bash
  python main.py --alg non-robust --training_steps 1 --max_iterations 1 --save_path exp1/ --env inventory
  ```
  ```bash
  python main.py --alg non-robust --training_steps 1 --max_iterations 1 --save_path exp1/ --env garnet
  ```
  ```bash
  python main.py --alg non-robust --training_steps 1 --max_iterations 1 --save_path exp1/ --env robot
  ```
