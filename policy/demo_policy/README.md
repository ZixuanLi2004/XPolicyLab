# demo_policy

**Contributor:** RoboDojo Team | **Paper:** Not applicable — XPolicyLab demo adapter | **arXiv:** Not applicable | **Original code:** Not applicable

`demo_policy` is the minimal reference adapter for checking policy-server / environment-client wiring. It returns two-step action chunks that hold the observed joint or end-effector posture and command zero base velocity when available; it does not train a real model or load a checkpoint. `process_data.sh` and `train.sh` are stubs that only demonstrate the standard naming conventions.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Installs XPolicyLab in editable mode so the policy server can import `XPolicyLab.policy.demo_policy` and the websocket modules:

```bash
cd XPolicyLab/policy/demo_policy
bash install.sh
conda activate <policy_env>  # e.g. demo-policy
```

## Data Processing

Stub — prints the standard output naming convention without converting data. The optional trailing argument is an episode-limit placeholder:

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

## Training

Stub — creates the conventional checkpoint directory `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` and prints where real training code should write:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

`demo_policy` does not read this directory during evaluation; real adapters load weights from their own checkpoint path.

## Evaluation

```bash
cd XPolicyLab/policy/demo_policy
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline shape/IO smoke check
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## MiVerse

The adapter accepts MiVerse runtime state dictionaries directly; it does not use a learned 45D or 75D model layout. Joint mode holds the arms, hands, waist and legs present in the observation. EE mode prefers `left_base_ee_pose` / `right_base_ee_pose`, holds `head_base_pose` when available, and lets whole-body IK control the waist and legs. Both modes command zero root linear and angular velocity when these fields are observed. Batch observations are matched by `additional_info.env_idx` or top-level `env_idx`, including subsets and reordered environments.

```bash
# No checkpoint is loaded; "demo" is a placeholder.
bash eval.sh MiVerse place_fruit_basket demo miverse joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

Use `ee` in place of `joint` to check the base-frame pose interface. This is an interface check, not a task-solving policy. Holding an EE target still runs IK and can produce small motion. The shared `EVAL_ENV_TYPE=debug` client checks transport and arm/hand shapes; its synthetic observations do not cover MiVerse waist, leg, head or base fields.

## Notes

- When replacing these stubs with a real policy, keep `ckpt_name` stable between data processing, training, and evaluation.
