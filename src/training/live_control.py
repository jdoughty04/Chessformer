"""
Live Training Controller

Provides real-time control over training hyperparameters via:
1. A JSON control file polled by the training loop
2. A lightweight web dashboard (no extra dependencies)
3. CLI tool compatibility (control_training.py)

Usage:
    controller = TrainingController(output_dir="checkpoints/run1", port=8585)
    controller.start()  # Starts web server in background thread
    
    # In training loop:
    changes = controller.poll()
    if changes:
        # Apply changes to optimizer, LoRA, etc.
"""

import json
import os
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any


# Default state written on first init
DEFAULT_STATE = {
    "lr_ratios": {
        "cnn_lr_ratio": 1.0,
        "transformer_lr_ratio": 10.0,
        "perceiver_lr_ratio": 10.0,
        "csmp_lr_ratio": 10.0,
        "xattn_lr_ratio": 1.0,
        "text_gate_lr_ratio": 1.0,
        "pseudotoken_lr_ratio": 1.0,
        "prepend_latent_lr_ratio": 1.0,
        "lora_lr_ratio": 1.0,
    },
    "aux_policy_weight": None,  # null = no change; float to apply
    "structured_xattn_sparse_weight": None,  # null = no change; float to apply
    "structured_xattn_square_diversity_weight": None,  # null = no change; float to apply
    "structured_xattn_square_diversity_target_entropy": None,  # null = no change; float to apply
    "aux_move_eval_weight": None,  # null = no change; float to apply
    "move_eval_mse_weight": None,  # null = no change; float to apply
    "move_eval_ce_weight": None,   # null = no change; float to apply
    "move_eval_pairwise_weight": None,   # null = no change; float to apply
    "bsr_weight": None,          # null = no change; float to apply
    "spp_weight": None,          # null = no change; float to apply
    "xattn_gate_tanh_value": None,  # null = no change; float in (-1, 1)
    "ffn_gate_tanh_value": None,  # null = no change; float in (-1, 1)
    "base_learning_rate": None,  # null = no change
    "commands": {
        "unfreeze_lora": False,
        "freeze_lora": False,
        "enable_lm": False,
        "disable_lm": False,
        "merge_and_reinit_lora": False,
        "rebuild_optimizer": False,
        "run_inference_sample": False,
        "freeze_cnn": False,
        "unfreeze_cnn": False,
        "freeze_transformer": False,
        "unfreeze_transformer": False,
        "freeze_csmp": False,
        "unfreeze_csmp": False,
        "freeze_perceiver": False,
        "unfreeze_perceiver": False,
        "freeze_xattn": False,
        "unfreeze_xattn": False,
        "freeze_prepend_latents": False,
        "unfreeze_prepend_latents": False,
        "freeze_lm_pseudotokens": False,
        "unfreeze_lm_pseudotokens": False,
    },
    "status": {
        "current_epoch": 0,
        "current_step": 0,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "lm_enabled": True,
        "lora_frozen": True,
        "cnn_frozen": False,
        "transformer_frozen": False,
        "csmp_frozen": False,
        "perceiver_frozen": False,
        "xattn_frozen": False,
        "prepend_latents_frozen": False,
        "lm_pseudotokens_frozen": False,
        "active_lr_ratios": {},
        "active_base_lr": 0.0,
        "active_aux_policy_weight": 0.1,
        "active_structured_xattn_sparse_weight": 0.0,
        "active_structured_xattn_square_diversity_weight": 0.0,
        "active_structured_xattn_square_diversity_target_entropy": 0.5,
        "active_aux_move_eval_weight": 0.0,
        "active_move_eval_mse_weight": 0.5,
        "active_move_eval_ce_weight": 0.5,
        "active_move_eval_pairwise_weight": 0.0,
        "active_bsr_weight": 0.0,
        "active_spp_weight": 0.0,
        "active_xattn_gate_tanh_mean": 0.0,
        "active_ffn_gate_tanh_mean": 0.0,
        "last_update": None,
        "last_command_applied": None,
        "inference_result": None,
    },
}

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Control Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:20px}
h1{font-size:1.5rem;margin-bottom:16px;color:#38bdf8}
h2{font-size:1.1rem;margin-bottom:10px;color:#94a3b8;border-bottom:1px solid #334155;padding-bottom:6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:900px;margin:0 auto}
.card{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155}
.card.full{grid-column:1/-1}
label{display:block;font-size:.85rem;color:#94a3b8;margin-bottom:4px}
input[type=number]{width:100%;padding:8px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:1rem;margin-bottom:10px}
input[type=number]:focus{outline:none;border-color:#38bdf8}
button{padding:8px 16px;border-radius:6px;border:none;cursor:pointer;font-weight:600;font-size:.9rem;transition:all .15s}
.btn-primary{background:#2563eb;color:#fff}.btn-primary:hover{background:#3b82f6}
.btn-warn{background:#d97706;color:#fff}.btn-warn:hover{background:#f59e0b}
.btn-danger{background:#dc2626;color:#fff}.btn-danger:hover{background:#ef4444}
.btn-success{background:#059669;color:#fff}.btn-success:hover{background:#10b981}
.btn-group{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.status-row{display:flex;justify-content:space-between;padding:4px 0;font-size:.9rem}
.status-val{color:#38bdf8;font-weight:600}
.toast{position:fixed;bottom:20px;right:20px;background:#059669;color:#fff;padding:12px 20px;border-radius:8px;font-weight:600;opacity:0;transition:opacity .3s;pointer-events:none}
.toast.show{opacity:1}
.toast.error{background:#dc2626}
#auto-refresh-label{font-size:.8rem;color:#64748b;margin-top:8px}
.inference-fen{font-family:monospace;font-size:.85rem;color:#cbd5e1;word-break:break-all;padding:8px;background:#0f172a;border-radius:6px;margin:6px 0}
.inference-text{font-size:.9rem;color:#e2e8f0;line-height:1.5;padding:10px;background:#0f172a;border-radius:6px;margin:6px 0;white-space:pre-wrap;max-height:300px;overflow-y:auto}
.inference-link{color:#38bdf8;text-decoration:none;font-weight:600;font-size:.85rem}.inference-link:hover{text-decoration:underline}
.inference-meta{font-size:.8rem;color:#64748b;margin-top:4px}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #475569;border-top-color:#38bdf8;border-radius:50%;animation:spin .6s linear infinite;margin-left:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.aux-panel{margin:8px 0;padding:8px;background:#0f172a;border-radius:6px;border:1px solid #1e293b}
.aux-section{margin:4px 0;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.aux-label{font-size:.78rem;font-weight:700;color:#94a3b8;min-width:90px;flex-shrink:0}
.aux-move{display:inline-block;padding:2px 8px;background:#1e293b;border-radius:4px;font-family:monospace;font-size:.85rem;color:#e2e8f0}
.aux-move small{color:#38bdf8;margin-left:2px}
.aux-move.maia{border:1px solid #475569}
.aux-move.maia small{color:#a78bfa}
.eval-bar{display:flex;height:22px;border-radius:4px;overflow:hidden;flex:1;min-width:120px}
.eval-seg{display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden}
</style>
</head>
<body>
<div class="grid">
<div class="card full"><h1>Training Live Control</h1></div>

<div class="card">
<h2>Learning Rate Ratios</h2>
<label>CNN LR Ratio</label>
<input type="number" id="cnn_lr_ratio" step="0.1" min="0">
<label>Transformer LR Ratio</label>
<input type="number" id="transformer_lr_ratio" step="0.1" min="0">
<label>Perceiver LR Ratio</label>
<input type="number" id="perceiver_lr_ratio" step="0.1" min="0">
<label>CSMP LR Ratio</label>
<input type="number" id="csmp_lr_ratio" step="0.1" min="0">
<label>X-Attn LR Ratio</label>
<input type="number" id="xattn_lr_ratio" step="0.1" min="0">
<label>Text Gate LR Ratio</label>
<input type="number" id="text_gate_lr_ratio" step="0.1" min="0">
<label>Pseudotoken LR Ratio</label>
<input type="number" id="pseudotoken_lr_ratio" step="0.1" min="0">
<label>Prepend Latent LR Ratio</label>
<input type="number" id="prepend_latent_lr_ratio" step="0.1" min="0">
<label>LoRA LR Ratio</label>
<input type="number" id="lora_lr_ratio" step="0.1" min="0">
<label>Base Learning Rate (leave 0 for no change)</label>
<input type="number" id="base_lr" step="0.000001" min="0" value="0">
<div class="btn-group">
<button class="btn-primary" onclick="applyLR()">Apply LR Changes</button>
</div>
</div>

<div class="card">
<h2>X-Attn Gate Control</h2>
<label>X-Attn Gate Value (tanh-space, -1 to 1)</label>
<input type="number" id="xattn_gate_tanh_value" step="0.05" min="-0.999" max="0.999" value="0">
<div class="btn-group">
<button class="btn-primary" onclick="applyXAttnGate()">Apply X-Attn Gate</button>
</div>
</div>

<div class="card">
<h2>FFN Gate Control</h2>
<label>FFN Gate Value (tanh-space, -1 to 1)</label>
<input type="number" id="ffn_gate_tanh_value" step="0.05" min="-0.999" max="0.999" value="0">
<div class="btn-group">
<button class="btn-primary" onclick="applyFFNGate()">Apply FFN Gate</button>
</div>
</div>

<div class="card">
<h2>Auxiliary Loss Weights</h2>
<label>Policy Distillation Weight</label>
<input type="number" id="aux_policy_weight" step="0.01" min="0" value="0.1">
<label>Structured X-Attn Sparse Weight</label>
<input type="number" id="structured_xattn_sparse_weight" step="0.01" min="0" value="0">
<label>Structured X-Attn Square Diversity Weight</label>
<input type="number" id="structured_xattn_square_diversity_weight" step="0.01" min="0" value="0">
<label>Structured X-Attn Square Diversity Target Entropy</label>
<input type="number" id="structured_xattn_square_diversity_target_entropy" step="0.01" min="0" max="1" value="0.5">
<label>Move Eval Weight (outer)</label>
<input type="number" id="aux_move_eval_weight" step="0.01" min="0" value="0">
<label>Move Eval MSE Weight (inner)</label>
<input type="number" id="move_eval_mse_weight" step="0.01" min="0" value="0.5">
<label>Move Eval CE Weight (inner)</label>
<input type="number" id="move_eval_ce_weight" step="0.01" min="0" value="0.5">
<label>Move Eval Pairwise Weight (inner)</label>
<input type="number" id="move_eval_pairwise_weight" step="0.01" min="0" value="0">
<label>BSR Weight (Board State Reconstruction)</label>
<input type="number" id="bsr_weight" step="0.01" min="0" value="0">
<label>SPP Weight (Square Property Prediction)</label>
<input type="number" id="spp_weight" step="0.01" min="0" value="0">
<div class="btn-group">
<button class="btn-primary" onclick="applyAuxWeights()">Apply Aux Weights</button>
</div>
</div>

<div class="card">
<h2>Training Status</h2>
<div id="status-panel">Loading...</div>
<div id="auto-refresh-label">Auto-refreshes every 5s</div>
</div>

<div class="card full">
<h2>LoRA Commands</h2>
<div class="btn-group">
<button class="btn-success" onclick="sendCmd('enable_lm')">Enable LM</button>
<button class="btn-warn" onclick="sendCmd('disable_lm')">Disable LM</button>
<button class="btn-success" onclick="sendCmd('unfreeze_lora')">Unfreeze LoRA</button>
<button class="btn-warn" onclick="sendCmd('freeze_lora')">Freeze LoRA</button>
<button class="btn-danger" onclick="sendCmd('merge_and_reinit_lora')">Merge &amp; Reinit LoRA</button>
</div>
</div>

<div class="card full">
<h2>Backbone Freeze Control</h2>
<div class="btn-group">
<button class="btn-warn" onclick="sendCmd('freeze_cnn')">Freeze CNN</button>
<button class="btn-success" onclick="sendCmd('unfreeze_cnn')">Unfreeze CNN</button>
<button class="btn-warn" onclick="sendCmd('freeze_transformer')">Freeze Transformer</button>
<button class="btn-success" onclick="sendCmd('unfreeze_transformer')">Unfreeze Transformer</button>
<button class="btn-warn" onclick="sendCmd('freeze_csmp')">Freeze CSMP</button>
<button class="btn-success" onclick="sendCmd('unfreeze_csmp')">Unfreeze CSMP</button>
<button class="btn-warn" onclick="sendCmd('freeze_perceiver')">Freeze Perceiver</button>
<button class="btn-success" onclick="sendCmd('unfreeze_perceiver')">Unfreeze Perceiver</button>
<button class="btn-warn" onclick="sendCmd('freeze_xattn')">Freeze X-Attn</button>
<button class="btn-success" onclick="sendCmd('unfreeze_xattn')">Unfreeze X-Attn</button>
<button class="btn-warn" onclick="sendCmd('freeze_prepend_latents')">Freeze Prepend Latents</button>
<button class="btn-success" onclick="sendCmd('unfreeze_prepend_latents')">Unfreeze Prepend Latents</button>
<button class="btn-warn" onclick="sendCmd('freeze_lm_pseudotokens')">Freeze LM Pseudotokens</button>
<button class="btn-success" onclick="sendCmd('unfreeze_lm_pseudotokens')">Unfreeze LM Pseudotokens</button>
<button class="btn-primary" onclick="sendCmd('rebuild_optimizer')">Rebuild Optimizer</button>
</div>
</div>

<div class="card full">
<h2>Live Inference</h2>
<div class="btn-group">
<button class="btn-primary" id="inference-btn" onclick="runInference()">Run Inference on Val Sample</button>
</div>
<div id="inference-panel" style="margin-top:12px"></div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
let initialized=false;
function toast(msg, err){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast'+(err?' error':'')+' show';
  setTimeout(()=>t.className='toast',2000);
}

async function fetchState(){
  try{
    const r=await fetch('/api/state');const d=await r.json();
    const keys=['cnn_lr_ratio','transformer_lr_ratio','perceiver_lr_ratio','csmp_lr_ratio','xattn_lr_ratio','text_gate_lr_ratio','pseudotoken_lr_ratio','prepend_latent_lr_ratio','lora_lr_ratio'];
    // Only populate inputs on first load
    if(!initialized){
      for(const k of keys){
        const el=document.getElementById(k);
        if(el) el.value=d.lr_ratios[k]||0;
      }
      const pw=document.getElementById('aux_policy_weight');
      if(pw && d.status && d.status.active_aux_policy_weight!==undefined) pw.value=d.status.active_aux_policy_weight;
      const sxw=document.getElementById('structured_xattn_sparse_weight');
      if(sxw && d.status && d.status.active_structured_xattn_sparse_weight!==undefined) sxw.value=d.status.active_structured_xattn_sparse_weight;
      const sxdw=document.getElementById('structured_xattn_square_diversity_weight');
      if(sxdw && d.status && d.status.active_structured_xattn_square_diversity_weight!==undefined) sxdw.value=d.status.active_structured_xattn_square_diversity_weight;
      const sxdt=document.getElementById('structured_xattn_square_diversity_target_entropy');
      if(sxdt && d.status && d.status.active_structured_xattn_square_diversity_target_entropy!==undefined) sxdt.value=d.status.active_structured_xattn_square_diversity_target_entropy;
      const mew=document.getElementById('aux_move_eval_weight');
      if(mew && d.status && d.status.active_aux_move_eval_weight!==undefined) mew.value=d.status.active_aux_move_eval_weight;
      const mmw=document.getElementById('move_eval_mse_weight');
      if(mmw && d.status && d.status.active_move_eval_mse_weight!==undefined) mmw.value=d.status.active_move_eval_mse_weight;
      const mcw=document.getElementById('move_eval_ce_weight');
      if(mcw && d.status && d.status.active_move_eval_ce_weight!==undefined) mcw.value=d.status.active_move_eval_ce_weight;
      const mpw=document.getElementById('move_eval_pairwise_weight');
      if(mpw && d.status && d.status.active_move_eval_pairwise_weight!==undefined) mpw.value=d.status.active_move_eval_pairwise_weight;
      const bw=document.getElementById('bsr_weight');
      if(bw && d.status && d.status.active_bsr_weight!==undefined) bw.value=d.status.active_bsr_weight;
      const sw=document.getElementById('spp_weight');
      if(sw && d.status && d.status.active_spp_weight!==undefined) sw.value=d.status.active_spp_weight;
      const xg=document.getElementById('xattn_gate_tanh_value');
      if(xg && d.status && d.status.active_xattn_gate_tanh_mean!==undefined) xg.value=d.status.active_xattn_gate_tanh_mean;
      const fg=document.getElementById('ffn_gate_tanh_value');
      if(fg && d.status && d.status.active_ffn_gate_tanh_mean!==undefined) fg.value=d.status.active_ffn_gate_tanh_mean;
      initialized=true;
    }
    // Status panel (always updates)
    const s=d.status||{};
    let html='';
    const rows=[
      ['Epoch',s.current_epoch],['Step',s.current_step],
      ['Train Loss',(s.train_loss||0).toFixed(4)],['Val Loss',(s.val_loss||0).toFixed(4)],
      ['LM Enabled',s.lm_enabled?'Yes':'No'],
      ['LoRA Frozen',s.lora_frozen?'Yes':'No'],
      ['CNN Frozen',s.cnn_frozen?'Yes':'No'],
      ['Transformer Frozen',s.transformer_frozen?'Yes':'No'],
      ['CSMP Frozen',s.csmp_frozen?'Yes':'No'],
      ['Perceiver Frozen',s.perceiver_frozen?'Yes':'No'],
      ['X-Attn Frozen',s.xattn_frozen?'Yes':'No'],
      ['Prepend Latents Frozen',s.prepend_latents_frozen?'Yes':'No'],
      ['LM Pseudotokens Frozen',s.lm_pseudotokens_frozen?'Yes':'No'],
      ['Base LR',(s.active_base_lr||0).toExponential(2)],
      ['Policy Weight',s.active_aux_policy_weight],
      ['Structured X-Attn Sparse Weight',s.active_structured_xattn_sparse_weight],
      ['Structured X-Attn Square Diversity Weight',s.active_structured_xattn_square_diversity_weight],
      ['Structured X-Attn Square Diversity Target',s.active_structured_xattn_square_diversity_target_entropy],
      ['Move Eval Weight',s.active_aux_move_eval_weight],
      ['Move Eval MSE Weight',s.active_move_eval_mse_weight],
      ['Move Eval CE Weight',s.active_move_eval_ce_weight],
      ['Move Eval Pairwise Weight',s.active_move_eval_pairwise_weight],
      ['BSR Weight',s.active_bsr_weight],
      ['SPP Weight',s.active_spp_weight],
      ['X-Attn Gate Mean (tanh)',s.active_xattn_gate_tanh_mean],
      ['FFN Gate Mean (tanh)',s.active_ffn_gate_tanh_mean],
      ['Last Command',s.last_command_applied||'\u2014'],
      ['Last Update',s.last_update||'\u2014'],
    ];
    for(const[k,v] of rows) html+=`<div class="status-row"><span>${k}</span><span class="status-val">${v}</span></div>`;
    // Active ratios (what the optimizer is actually using)
    const ar=s.active_lr_ratios||{};
    for(const[k,v] of Object.entries(ar)) html+=`<div class="status-row"><span>Active ${k}</span><span class="status-val">x${v}</span></div>`;
    // Queued ratios (what's in the control file, pending next poll)
    const qr=d.lr_ratios||{};
    for(const k of keys){
      const active=ar[k], queued=qr[k];
      if(active!==undefined && queued!==undefined && active!==queued)
        html+=`<div class="status-row"><span>Queued ${k}</span><span class="status-val" style="color:#f59e0b">x${queued}</span></div>`;
    }
    document.getElementById('status-panel').innerHTML=html;
    // Inference result panel
    renderInference(s.inference_result);
  }catch(e){console.error(e)}
}

async function applyLR(){
  const body={lr_ratios:{}};
  for(const k of ['cnn_lr_ratio','transformer_lr_ratio','perceiver_lr_ratio','csmp_lr_ratio','xattn_lr_ratio','text_gate_lr_ratio','pseudotoken_lr_ratio','prepend_latent_lr_ratio','lora_lr_ratio']){
    body.lr_ratios[k]=parseFloat(document.getElementById(k).value)||0;
  }
  const blr=parseFloat(document.getElementById('base_lr').value);
  if(blr>0) body.base_learning_rate=blr;
  body.commands={rebuild_optimizer:true};
  try{
    const r=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok) toast('LR changes queued!'); else toast('Failed','err');
  }catch(e){toast('Error: '+e,'err')}
}

async function applyAuxWeights(){
  const pw=parseFloat(document.getElementById('aux_policy_weight').value);
  const sxw=parseFloat(document.getElementById('structured_xattn_sparse_weight').value);
  const sxdw=parseFloat(document.getElementById('structured_xattn_square_diversity_weight').value);
  const sxdt=parseFloat(document.getElementById('structured_xattn_square_diversity_target_entropy').value);
  const mew=parseFloat(document.getElementById('aux_move_eval_weight').value);
  const mmw=parseFloat(document.getElementById('move_eval_mse_weight').value);
  const mcw=parseFloat(document.getElementById('move_eval_ce_weight').value);
  const mpw=parseFloat(document.getElementById('move_eval_pairwise_weight').value);
  const bw=parseFloat(document.getElementById('bsr_weight').value);
  const sw=parseFloat(document.getElementById('spp_weight').value);
  const body={};
  if(!isNaN(pw)&&pw>=0) body.aux_policy_weight=pw;
  if(!isNaN(sxw)&&sxw>=0) body.structured_xattn_sparse_weight=sxw;
  if(!isNaN(sxdw)&&sxdw>=0) body.structured_xattn_square_diversity_weight=sxdw;
  if(!isNaN(sxdt)&&sxdt>=0&&sxdt<=1) body.structured_xattn_square_diversity_target_entropy=sxdt;
  if(!isNaN(mew)&&mew>=0) body.aux_move_eval_weight=mew;
  if(!isNaN(mmw)&&mmw>=0) body.move_eval_mse_weight=mmw;
  if(!isNaN(mcw)&&mcw>=0) body.move_eval_ce_weight=mcw;
  if(!isNaN(mpw)&&mpw>=0) body.move_eval_pairwise_weight=mpw;
  if(!isNaN(bw)&&bw>=0) body.bsr_weight=bw;
  if(!isNaN(sw)&&sw>=0) body.spp_weight=sw;
  if(!Object.keys(body).length){toast('No valid weights','err');return;}
  try{
    const r=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok) toast('Aux weights queued'); else toast('Failed','err');
  }catch(e){toast('Error: '+e,'err')}
}

async function applyXAttnGate(){
  const gv=parseFloat(document.getElementById('xattn_gate_tanh_value').value);
  if(isNaN(gv)){toast('Invalid x-attn gate value','err');return;}
  if(gv<=-1||gv>=1){toast('Gate must be in (-1, 1)','err');return;}
  const body={xattn_gate_tanh_value:gv};
  try{
    const r=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok) toast('X-Attn gate update queued'); else toast('Failed','err');
  }catch(e){toast('Error: '+e,'err')}
}

async function applyFFNGate(){
  const gv=parseFloat(document.getElementById('ffn_gate_tanh_value').value);
  if(isNaN(gv)){toast('Invalid FFN gate value','err');return;}
  if(gv<=-1||gv>=1){toast('Gate must be in (-1, 1)','err');return;}
  const body={ffn_gate_tanh_value:gv};
  try{
    const r=await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(r.ok) toast('FFN gate update queued'); else toast('Failed','err');
  }catch(e){toast('Error: '+e,'err')}
}

async function sendCmd(cmd){
  try{
    const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:cmd})});
    if(r.ok) toast('Command queued: '+cmd); else toast('Failed','err');
  }catch(e){toast('Error: '+e,'err')}
}

let inferenceWaiting=false;
async function runInference(){
  const btn=document.getElementById('inference-btn');
  btn.disabled=true;btn.innerHTML='Running...<span class="spinner"></span>';
  inferenceWaiting=true;
  await sendCmd('run_inference_sample');
}

function renderInference(result){
  const panel=document.getElementById('inference-panel');
  if(!result){panel.innerHTML='';return;}
  const step=result.step||'?';
  const ts=result.timestamp||'';
  let html=`<div class="inference-meta">Step ${step} &middot; ${ts}</div>`;
  if(result.error){
    html+=`<div class="inference-text" style="color:#f87171">Error: ${result.error}</div>`;
  }else{
    const fen=result.fen||'';
    const commentary=result.commentary||'(empty)';
    const lichessUrl='https://lichess.org/analysis/'+encodeURIComponent(fen).replace(/%20/g,'_');
    html+=`<div class="inference-fen">${fen}</div>`;
    html+=`<a class="inference-link" href="${lichessUrl}" target="_blank" rel="noopener">View on Lichess Board &rarr;</a>`;
    // Auxiliary predictions
    const aux=result.aux;
    if(aux){
      html+=`<div class="aux-panel">`;
      if(aux.num_legal_moves!==undefined){
        const term=aux.position_terminal?' (terminal)':'';
        html+=`<div class="aux-section"><span class="aux-label">Legality</span>`;
        html+=`<span class="aux-move">Legal moves: ${aux.num_legal_moves}${term}</span>`;
        if(aux.top_moves_illegal_in_raw_top5!==undefined)
          html+=`<span class="aux-move">Adapter raw illegal: ${aux.top_moves_illegal_in_raw_top5}/5</span>`;
        if(aux.maia_top_moves_illegal_in_raw_top5!==undefined)
          html+=`<span class="aux-move maia">Maia raw illegal: ${aux.maia_top_moves_illegal_in_raw_top5}/5</span>`;
        html+=`</div>`;
      }
      if(aux.legality_unavailable){
        const err=aux.legality_error||'unknown error';
        html+=`<div class="aux-section"><span class="aux-label">Legality</span>`;
        html+=`<span class="aux-move">Unavailable: ${err}</span>`;
        html+=`</div>`;
      }
      if(aux.top_moves){
        html+=`<div class="aux-section"><span class="aux-label">Adapter Policy</span>`;
        if(aux.top_moves.length===0)
          html+=`<span class="aux-move">No legal move found in policy distribution.</span>`;
        else
          html+=aux.top_moves.map(m=>`<span class="aux-move">${m.move} <small>${m.prob}%</small></span>`).join('');
        html+=`</div>`;
      }
      if(aux.maia_top_moves){
        html+=`<div class="aux-section"><span class="aux-label">Maia Target</span>`;
        if(aux.maia_top_moves.length===0)
          html+=`<span class="aux-move maia">${aux.maia_target_unavailable_reason||'No legal move found in Maia target distribution.'}</span>`;
        else
          html+=aux.maia_top_moves.map(m=>`<span class="aux-move maia">${m.move} <small>${m.prob}%</small></span>`).join('');
        html+=`</div>`;
      }
      if(aux.move_eval_topk){
        html+=`<div class="aux-section" style="flex-direction:column;align-items:flex-start">`;
        html+=`<span class="aux-label">Move Eval (CE top-k)</span>`;
        if(aux.move_eval_topk.length===0){
          html+=`<span class="aux-move">${aux.move_eval_unavailable_reason||'No supervised CE candidates.'}</span>`;
        }else{
          html+=`<div style="display:flex;gap:6px;flex-wrap:wrap">`+
            aux.move_eval_topk.map(m=>`<span class="aux-move">${m.move} <small>tgt ${m.target_cp}cp | pred ${m.pred_cp}cp | tgtP ${m.target_prob}% | polP ${m.policy_prob}%</small></span>`).join('')+
            `</div>`;
        }
        html+=`</div>`;
      }
      if(aux.move_eval_policy_top5){
        html+=`<div class="aux-section" style="flex-direction:column;align-items:flex-start">`;
        html+=`<span class="aux-label">Move Eval (Policy top-5)</span>`;
        if(aux.move_eval_policy_top5.length===0){
          html+=`<span class="aux-move">Unavailable.</span>`;
        }else{
          html+=`<div style="display:flex;gap:6px;flex-wrap:wrap">`+
            aux.move_eval_policy_top5.map(m=>`<span class="aux-move">${m.move} <small>pred ${m.pred_cp}cp | polP ${m.policy_prob}%</small></span>`).join('')+
            `</div>`;
        }
        html+=`</div>`;
      }
      if(aux.bsr_board){
        html+=`<div class="aux-section" style="flex-direction:column;align-items:flex-start">`;
        html+=`<span class="aux-label">BSR (Board Reconstruction)${aux.bsr_accuracy!==undefined?' — '+aux.bsr_accuracy+'% acc':''}</span>`;
        html+=`<div style="display:flex;gap:16px;margin-top:4px">`;
        html+=`<div><div style="font-size:.7rem;color:#64748b;margin-bottom:2px">Predicted</div><pre style="font-family:monospace;font-size:.85rem;color:#e2e8f0;background:#0f172a;padding:6px 8px;border-radius:4px;line-height:1.4;margin:0">${aux.bsr_board.join('\\n')}</pre></div>`;
        if(aux.bsr_gt_board){
          html+=`<div><div style="font-size:.7rem;color:#64748b;margin-bottom:2px">Ground Truth</div><pre style="font-family:monospace;font-size:.85rem;color:#e2e8f0;background:#0f172a;padding:6px 8px;border-radius:4px;line-height:1.4;margin:0">${aux.bsr_gt_board.join('\\n')}</pre></div>`;
        }
        html+=`</div></div>`;
      }
      if(aux.spp_summary){
        const s=aux.spp_summary;
        html+=`<div class="aux-section"><span class="aux-label">SPP</span>`;
        html+=`<span class="aux-move">W⚔${s.white_attacks}</span>`;
        html+=`<span class="aux-move">B⚔${s.black_attacks}</span>`;
        html+=`<span class="aux-move">Ray ${s.avg_ray_dist}</span>`;
        if(aux.spp_gt_summary){
          const g=aux.spp_gt_summary;
          html+=`<span style="color:#64748b;font-size:.75rem;margin-left:4px">GT: W⚔${g.white_attacks} B⚔${g.black_attacks} Ray ${g.avg_ray_dist}</span>`;
        }
        html+=`</div>`;
      }
      html+=`</div>`;
    }
    html+=`<div class="inference-text">${commentary.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
  }
  panel.innerHTML=html;
  if(inferenceWaiting){
    inferenceWaiting=false;
    const btn=document.getElementById('inference-btn');
    btn.disabled=false;btn.innerHTML='Run Inference on Val Sample';
    toast(result.error?'Inference failed':'Inference complete!',!!result.error);
  }
}

fetchState();setInterval(fetchState,5000);
</script>
</body>
</html>"""


class TrainingController:
    """Manages a JSON control file and optional web dashboard for live training control."""

    def __init__(self, output_dir: str, port: int = 8585, poll_steps: int = 10):
        self.output_dir = Path(output_dir)
        self.control_file = self.output_dir / "training_control.json"
        self.port = port
        self.poll_steps = poll_steps
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_mtime = 0.0
        self._pending_changes: Dict[str, Any] = {}

    # ── File I/O ──────────────────────────────────────────────

    def _read_state(self) -> dict:
        try:
            with open(self.control_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_state(self, state: dict):
        tmp = str(self.control_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(self.control_file))

    def init_control_file(self, config) -> dict:
        """Initialize control file from training config. Returns the initial state."""
        state = json.loads(json.dumps(DEFAULT_STATE))  # deep copy
        # Read LR ratios from the active config section
        mode = getattr(config.model, "mode", "")
        if mode in ("chess_fusion", "policy_only"):
            src = getattr(config.model, "chess_fusion", None)
        else:
            src = getattr(config.model, "maia", None)
        if src:
            state["lr_ratios"]["cnn_lr_ratio"] = getattr(src, "cnn_lr_ratio", 0.1)
            state["lr_ratios"]["transformer_lr_ratio"] = getattr(src, "transformer_lr_ratio", 0.1)
            state["lr_ratios"]["perceiver_lr_ratio"] = getattr(src, "perceiver_lr_ratio", 1.0)
            state["lr_ratios"]["csmp_lr_ratio"] = getattr(src, "csmp_lr_ratio", None)
            if state["lr_ratios"]["csmp_lr_ratio"] is None:
                state["lr_ratios"]["csmp_lr_ratio"] = state["lr_ratios"]["perceiver_lr_ratio"]
            state["lr_ratios"]["lora_lr_ratio"] = getattr(src, "lora_lr_ratio", 1.0)
            state["lr_ratios"]["xattn_lr_ratio"] = getattr(src, "xattn_lr_ratio", 1.0)
            state["lr_ratios"]["text_gate_lr_ratio"] = getattr(src, "text_gate_lr_ratio", None)
            if state["lr_ratios"]["text_gate_lr_ratio"] is None:
                state["lr_ratios"]["text_gate_lr_ratio"] = state["lr_ratios"]["xattn_lr_ratio"]
            state["lr_ratios"]["pseudotoken_lr_ratio"] = getattr(src, "pseudotoken_lr_ratio", None)
            if state["lr_ratios"]["pseudotoken_lr_ratio"] is None:
                state["lr_ratios"]["pseudotoken_lr_ratio"] = state["lr_ratios"]["xattn_lr_ratio"]
            state["lr_ratios"]["prepend_latent_lr_ratio"] = getattr(src, "prepend_latent_lr_ratio", None)
            if state["lr_ratios"]["prepend_latent_lr_ratio"] is None:
                state["lr_ratios"]["prepend_latent_lr_ratio"] = state["lr_ratios"]["xattn_lr_ratio"]
        state["base_learning_rate"] = None
        state["aux_policy_weight"] = None
        state["structured_xattn_sparse_weight"] = None
        state["structured_xattn_square_diversity_weight"] = None
        state["structured_xattn_square_diversity_target_entropy"] = None
        state["aux_move_eval_weight"] = None
        state["move_eval_mse_weight"] = None
        state["move_eval_ce_weight"] = None
        state["move_eval_pairwise_weight"] = None
        state["bsr_weight"] = None
        state["spp_weight"] = None
        state["xattn_gate_tanh_value"] = None
        state["ffn_gate_tanh_value"] = None
        state["status"]["lm_enabled"] = bool(mode != "policy_only" and getattr(config.model, "enable_lm", True))
        state["status"]["active_base_lr"] = config.learning_rate
        if src:
            state["status"]["active_aux_policy_weight"] = getattr(src, "aux_policy_weight", 0.1)
            state["status"]["active_structured_xattn_sparse_weight"] = getattr(src, "structured_xattn_sparse_weight", 0.0)
            state["status"]["active_structured_xattn_square_diversity_weight"] = getattr(src, "structured_xattn_square_diversity_weight", 0.0)
            state["status"]["active_structured_xattn_square_diversity_target_entropy"] = getattr(src, "structured_xattn_square_diversity_target_entropy", 0.5)
            state["status"]["active_aux_move_eval_weight"] = getattr(src, "aux_move_eval_weight", 0.0)
            state["status"]["active_move_eval_mse_weight"] = getattr(src, "move_eval_mse_weight", 0.5)
            state["status"]["active_move_eval_ce_weight"] = getattr(src, "move_eval_ce_weight", 0.5)
            state["status"]["active_move_eval_pairwise_weight"] = getattr(src, "move_eval_pairwise_weight", 0.0)
            state["status"]["active_bsr_weight"] = getattr(src, "bsr_weight", 0.0)
            state["status"]["active_spp_weight"] = getattr(src, "spp_weight", 0.0)
            state["status"]["active_xattn_gate_tanh_mean"] = float(getattr(src, "xattn_gate_init", 0.0))
            state["status"]["active_ffn_gate_tanh_mean"] = float(getattr(src, "xattn_gate_init", 0.0))
        state["status"]["active_lr_ratios"] = dict(state["lr_ratios"])
        if src:
            state["status"]["xattn_frozen"] = getattr(src, "freeze_xattn", False)
            state["status"]["cnn_frozen"] = getattr(src, "freeze_cnn", True)
            state["status"]["transformer_frozen"] = getattr(src, "freeze_transformer", True)
            state["status"]["csmp_frozen"] = getattr(src, "freeze_csmp", False)
            state["status"]["perceiver_frozen"] = getattr(src, "freeze_perceiver", False)
            state["status"]["prepend_latents_frozen"] = getattr(src, "freeze_prepend_latents", False)
            state["status"]["lm_pseudotokens_frozen"] = getattr(src, "freeze_lm_pseudotokens", False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_state(state)
        self._last_mtime = os.path.getmtime(self.control_file)
        return state

    # ── Polling ───────────────────────────────────────────────

    def poll(self) -> Optional[Dict[str, Any]]:
        """Check if control file has been modified. Returns dict of changes or None."""
        try:
            mtime = os.path.getmtime(self.control_file)
        except FileNotFoundError:
            return None
        if mtime <= self._last_mtime:
            return None
        with self._lock:
            self._last_mtime = mtime
            state = self._read_state()
            if not state:
                return None
            changes = {}

            # Check for LR ratio changes
            if "lr_ratios" in state:
                changes["lr_ratios"] = state["lr_ratios"]

            # Check for base LR change
            need_write = False
            if state.get("base_learning_rate") is not None:
                changes["base_learning_rate"] = state["base_learning_rate"]
                state["base_learning_rate"] = None  # Clear after reading
                need_write = True

            # Check for aux policy weight change
            if state.get("aux_policy_weight") is not None:
                changes["aux_policy_weight"] = state["aux_policy_weight"]
                state["aux_policy_weight"] = None  # Clear after reading
                need_write = True
            if state.get("structured_xattn_sparse_weight") is not None:
                changes["structured_xattn_sparse_weight"] = state["structured_xattn_sparse_weight"]
                state["structured_xattn_sparse_weight"] = None
                need_write = True
            if state.get("structured_xattn_square_diversity_weight") is not None:
                changes["structured_xattn_square_diversity_weight"] = state["structured_xattn_square_diversity_weight"]
                state["structured_xattn_square_diversity_weight"] = None
                need_write = True
            if state.get("structured_xattn_square_diversity_target_entropy") is not None:
                changes["structured_xattn_square_diversity_target_entropy"] = state["structured_xattn_square_diversity_target_entropy"]
                state["structured_xattn_square_diversity_target_entropy"] = None
                need_write = True

            # Check for move-eval weight changes
            if state.get("aux_move_eval_weight") is not None:
                changes["aux_move_eval_weight"] = state["aux_move_eval_weight"]
                state["aux_move_eval_weight"] = None
                need_write = True
            if state.get("move_eval_mse_weight") is not None:
                changes["move_eval_mse_weight"] = state["move_eval_mse_weight"]
                state["move_eval_mse_weight"] = None
                need_write = True
            if state.get("move_eval_ce_weight") is not None:
                changes["move_eval_ce_weight"] = state["move_eval_ce_weight"]
                state["move_eval_ce_weight"] = None
                need_write = True
            if state.get("move_eval_pairwise_weight") is not None:
                changes["move_eval_pairwise_weight"] = state["move_eval_pairwise_weight"]
                state["move_eval_pairwise_weight"] = None
                need_write = True

            # Check for BSR weight change
            if state.get("bsr_weight") is not None:
                changes["bsr_weight"] = state["bsr_weight"]
                state["bsr_weight"] = None
                need_write = True

            # Check for SPP weight change
            if state.get("spp_weight") is not None:
                changes["spp_weight"] = state["spp_weight"]
                state["spp_weight"] = None
                need_write = True

            # Check for x-attn gate value change
            if state.get("xattn_gate_tanh_value") is not None:
                changes["xattn_gate_tanh_value"] = state["xattn_gate_tanh_value"]
                state["xattn_gate_tanh_value"] = None
                need_write = True

            # Check for FFN gate value change
            if state.get("ffn_gate_tanh_value") is not None:
                changes["ffn_gate_tanh_value"] = state["ffn_gate_tanh_value"]
                state["ffn_gate_tanh_value"] = None
                need_write = True

            # Check for commands
            cmds = state.get("commands", {})
            active_cmds = {k: v for k, v in cmds.items() if v}
            if active_cmds:
                changes["commands"] = active_cmds
                # Clear commands after reading
                for k in active_cmds:
                    state["commands"][k] = False
                need_write = True

            if need_write:
                self._write_state(state)
                self._last_mtime = os.path.getmtime(self.control_file)

            return changes if changes else None

    def update_status(self, **kwargs):
        """Update the status section of the control file (called from training loop)."""
        with self._lock:
            state = self._read_state()
            if not state:
                return
            status = state.get("status", {})
            status.update(kwargs)
            status["last_update"] = time.strftime("%H:%M:%S")
            state["status"] = status
            self._write_state(state)
            # Only advance mtime tracking if no pending commands/changes,
            # otherwise poll() would skip them thinking nothing changed.
            has_pending = (state.get("base_learning_rate") is not None
                          or state.get("aux_policy_weight") is not None
                          or state.get("structured_xattn_sparse_weight") is not None
                          or state.get("structured_xattn_square_diversity_weight") is not None
                          or state.get("structured_xattn_square_diversity_target_entropy") is not None
                          or state.get("aux_move_eval_weight") is not None
                          or state.get("move_eval_mse_weight") is not None
                          or state.get("move_eval_ce_weight") is not None
                          or state.get("move_eval_pairwise_weight") is not None
                          or state.get("bsr_weight") is not None
                          or state.get("spp_weight") is not None
                          or state.get("xattn_gate_tanh_value") is not None
                          or state.get("ffn_gate_tanh_value") is not None)
            if not has_pending:
                cmds = state.get("commands", {})
                has_pending = any(v for v in cmds.values())
            if not has_pending:
                self._last_mtime = os.path.getmtime(self.control_file)

    # ── Web Dashboard ─────────────────────────────────────────

    def start(self, rank: int = 0):
        """Start web dashboard in a background thread (only on rank 0)."""
        if rank != 0:
            return
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress request logs

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode())
                elif self.path == "/api/state":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    state = controller._read_state()
                    self.wfile.write(json.dumps(state).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}

                if self.path == "/api/update":
                    with controller._lock:
                        state = controller._read_state()
                        if "lr_ratios" in body:
                            state["lr_ratios"].update(body["lr_ratios"])
                        if "base_learning_rate" in body:
                            state["base_learning_rate"] = body["base_learning_rate"]
                        if "aux_policy_weight" in body:
                            state["aux_policy_weight"] = body["aux_policy_weight"]
                        if "structured_xattn_sparse_weight" in body:
                            state["structured_xattn_sparse_weight"] = body["structured_xattn_sparse_weight"]
                        if "structured_xattn_square_diversity_weight" in body:
                            state["structured_xattn_square_diversity_weight"] = body["structured_xattn_square_diversity_weight"]
                        if "structured_xattn_square_diversity_target_entropy" in body:
                            state["structured_xattn_square_diversity_target_entropy"] = body["structured_xattn_square_diversity_target_entropy"]
                        if "aux_move_eval_weight" in body:
                            state["aux_move_eval_weight"] = body["aux_move_eval_weight"]
                        if "move_eval_mse_weight" in body:
                            state["move_eval_mse_weight"] = body["move_eval_mse_weight"]
                        if "move_eval_ce_weight" in body:
                            state["move_eval_ce_weight"] = body["move_eval_ce_weight"]
                        if "move_eval_pairwise_weight" in body:
                            state["move_eval_pairwise_weight"] = body["move_eval_pairwise_weight"]
                        if "bsr_weight" in body:
                            state["bsr_weight"] = body["bsr_weight"]
                        if "spp_weight" in body:
                            state["spp_weight"] = body["spp_weight"]
                        if "xattn_gate_tanh_value" in body:
                            state["xattn_gate_tanh_value"] = body["xattn_gate_tanh_value"]
                        if "ffn_gate_tanh_value" in body:
                            state["ffn_gate_tanh_value"] = body["ffn_gate_tanh_value"]
                        if "commands" in body:
                            for k, v in body["commands"].items():
                                state["commands"][k] = v
                        controller._write_state(state)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')

                elif self.path == "/api/command":
                    cmd = body.get("command")
                    if cmd:
                        with controller._lock:
                            state = controller._read_state()
                            state["commands"][cmd] = True
                            controller._write_state(state)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"ok":true}')
                    else:
                        self.send_response(400)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

        try:
            self._server = HTTPServer(("0.0.0.0", self.port), Handler)
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            print(f"\n[Live Control] Dashboard running at http://localhost:{self.port}")
            print(f"[Live Control] Control file: {self.control_file}")
            print(f"[Live Control] SSH tunnel: ssh -L {self.port}:localhost:{self.port} <host>")
        except OSError as e:
            print(f"\n[Live Control] Could not start web dashboard on port {self.port}: {e}")
            print(f"[Live Control] File-based control still active at: {self.control_file}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

