"""Experimental fine-tuning for low-amplitude load identification.

This augmentation is valid only if the strain input is the mechanical-strain
response and scaling the applied load scales that response at fixed thermal
field and structural parameters. It never uses test cases to update weights.
The frozen base checkpoint is retained unchanged for reproducibility.
"""

from Net_v3_Attention import BPPointNet, _read_training_data
import json
import pathlib
import sys

import numpy as np
import torch

project = pathlib.Path(__file__).resolve().parent
output = project/'Net_Attention_v3_amplitude_candidate.pth'
sys.path.insert(0, str(project))

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = torch.load(project/'Net_Attention_v3_base_for_amplitude.pth',
                  map_location='cpu', weights_only=True)
stats = {k: v.to(device) for k, v in base['normalization'].items()}
model = BPPointNet(**base['architecture']).to(device)
model.load_state_dict(base['model_state_dict'])
x, zt, zu, y = [a.to(device) for a in _read_training_data(project)]
tr = slice(0, 3200)
va = slice(3200, 3600)
te = slice(3600, 4000)
yrms = torch.sqrt(torch.mean(y[tr]**2))
small_scales = torch.tensor([.025, .05, .1, .25, .5], device=device)
low_weight = 0.03
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


def pred(x0, t0, u0):
    return model((x0-stats['x_mean'])/stats['x_std'],
                 (t0-stats['z1_mean'])/stats['z1_std'],
                 (u0-stats['z2_mean'])/stats['z2_std'])*stats['y_std']+stats['y_mean']


best = float('inf')
best_state = None
best_epoch = 0
wait = 0
epochs = 1500
for epoch in range(1, epochs + 1):
    model.train()
    opt.zero_grad()
    sc = small_scales[torch.randint(
        len(small_scales), (len(x[tr]), 1), device=device)]
    full = pred(x[tr], zt[tr], zu[tr])
    low = pred(x[tr]*sc, zt[tr], zu[tr])
    full_loss = torch.mean(((full-y[tr])/yrms)**2)
    low_loss = torch.mean(((low-y[tr]*sc)/(yrms*sc))**2)
    loss = full_loss+low_weight*low_loss
    loss.backward()
    opt.step()
    if epoch % 25 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            vf = pred(x[va], zt[va], zu[va])
            vfl = float(torch.mean(((vf-y[va])/yrms)**2))
            vls = []
            for s in small_scales:
                vl = pred(x[va]*s, zt[va], zu[va])
                vls.append(float(torch.mean(((vl-y[va]*s)/(yrms*s))**2)))
            vlm = float(np.mean(vls))
            score = vfl+low_weight*vlm
        if score < best-1e-8:
            best = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 25
        if epoch % 100 == 0 or epoch == 1:
            print(
                f'epoch={epoch} train={float(loss):.6f} val_full={vfl:.6f} val_low={vlm:.6f}', flush=True)
        if wait >= 3000:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    result = {'seed': 42, 'train_count': 3200, 'validation_count': 400, 'test_count': 400,
              'base_checkpoint_best_epoch': base['metrics']['best_epoch'],
              'best_epoch': best_epoch, 'validation_score': best,
              'low_weight': low_weight, 'learning_rate': 1e-4,
              'small_load_factors': [float(v) for v in small_scales.cpu()]}
    for label, s in [('full', 1.), ('low', .025)]:
        p = pred(x[te]*s, zt[te], zu[te])
        tar = y[te]*s
        result['test_'+label] = float(torch.linalg.vector_norm(p-tar) /
                                      torch.linalg.vector_norm(tar))
    d = project/'testdata'/'7'
    cx = torch.tensor(np.loadtxt(d/'strains.txt').T,
                      dtype=torch.float32, device=device)
    ct = torch.tensor(np.loadtxt(d/'Thermal_Coefs.txt'),
                      dtype=torch.float32, device=device)
    cu = torch.tensor(np.loadtxt(d/'uq_data.txt').T,
                      dtype=torch.float32, device=device)
    cy = torch.tensor(np.loadtxt(d/'F_data.txt').reshape(-1, 1),
                      dtype=torch.float32, device=device)
    cp = pred(cx, ct, cu)
    result['case7_raw'] = float(torch.linalg.vector_norm(
        cp-cy)/torch.linalg.vector_norm(cy))
    result['case7_shifted'] = float(torch.linalg.vector_norm(
        cp-cp[0]-cy)/torch.linalg.vector_norm(cy))
torch.save({'model_state_dict': best_state, 'normalization': base['normalization'],
            'architecture': base['architecture'], 'metrics': result}, output)
output.with_suffix('.json').write_text(
    json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
