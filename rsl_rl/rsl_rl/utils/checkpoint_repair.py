import argparse
import torch
import sys

def pick_key(d, primary, fallback):
    if primary in d:
        return primary
    if fallback in d:
        return fallback
    raise KeyError(f"Neither '{primary}' nor '{fallback}' found in obs_rms_state_dict. Available keys: {list(d.keys())}")

def main():
    ap = argparse.ArgumentParser(description='Map old obs_rms_state_dict -> actor_obs_rms & critic_obs_rms inside model_state_dict.')
    ap.add_argument('--in',  dest='inp', required=True, help='Path to old checkpoint (.pt)')
    ap.add_argument('--out', dest='out', required=True, help='Path to write patched checkpoint')
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location='cpu')

    if 'model_state_dict' not in ckpt:
        print("ERROR: checkpoint missing 'model_state_dict'.", file=sys.stderr)
        sys.exit(2)
    if 'obs_rms_state_dict' not in ckpt or ckpt['obs_rms_state_dict'] is None:
        print("ERROR: checkpoint missing 'obs_rms_state_dict' (old obs_mean_std). This patcher only handles that case.", file=sys.stderr)
        sys.exit(3)

    sd = ckpt['model_state_dict']
    obs_blob = ckpt['obs_rms_state_dict']

    # Support both naming conventions: running_mean/var or mean/var
    rm_key = pick_key(obs_blob, 'running_mean', 'mean')
    rv_key = pick_key(obs_blob, 'running_var', 'var')
    ct_key = 'count'
    if ct_key not in obs_blob:
        print("ERROR: 'count' not found in obs_rms_state_dict.", file=sys.stderr)
        sys.exit(4)

    rm = obs_blob[rm_key].detach().clone()
    rv = obs_blob[rv_key].detach().clone()
    ct = obs_blob[ct_key].detach().clone()

    # Inject into new locations
    for prefix in ['actor_obs_rms', 'critic_obs_rms']:
        sd[f'{prefix}.running_mean'] = rm
        sd[f'{prefix}.running_var']  = rv
        sd[f'{prefix}.count']        = ct

    ckpt['model_state_dict'] = sd

    torch.save(ckpt, args.out)
    print(f"Patched checkpoint written to: {args.out}")
    print("Mapped obs_rms_state_dict -> actor_obs_rms & critic_obs_rms. Done.")

if __name__ == '__main__':
    main()