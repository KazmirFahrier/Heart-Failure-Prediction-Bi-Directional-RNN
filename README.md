<h1>Heart Failure Prediction — Bi-Directional RNN (HW3)</h1>

<p>
This project builds a <strong>bi-directional RNN</strong> over sequences of diagnosis labels (per-visit ICD-9) to predict Heart Failure.
It includes dataset preparation (padding + masks + reversed sequences), vectorized masking ops, a GRU-based model,
and a full training/evaluation loop with ROC-AUC reporting.
</p>

<hr/>

<h2>📁 Project Layout</h2>
<pre><code>.
├─ HW3_RNN.ipynb
├─ HW3_RNN-lib/
│  └─ data/
│     ├─ train/
│     │  ├─ pids.pkl   ├─ vids.pkl   ├─ hfs.pkl
│     │  ├─ seqs.pkl   ├─ types.pkl  └─ rtypes.pkl
│     └─ val/ (built by split in the notebook)
└─ README.md
</code></pre>

<hr/>

<h2>🧠 Problem &amp; Data</h2>
<ul>
  <li><strong>Goal:</strong> Predict whether a patient develops HF using sequences of diagnosis labels across visits.</li>
  <li><strong>Inputs:</strong> <code>seqs[i][j][k]</code> = k-th diagnosis label at the j-th visit for the i-th patient.</li>
  <li><strong>Labels:</strong> <code>hfs[i] ∈ {0,1}</code> (0 = non-HF, 1 = HF).</li>
  <li><strong>Size:</strong> 1,000 patients; label space size 619.</li>
</ul>

<hr/>

<h2>🚦 Pipeline</h2>

<h3>1) Dataset &amp; DataLoader</h3>
<ol>
  <li><strong>CustomDataset</strong>: stores raw <code>(sequences, labels)</code>; no tensor conversion here.</li>
  <li><strong>collate_fn</strong> (batching):
    <ul>
      <li>Pad to a uniform 3-D tensor: <code>x ∈ ℤ^{B×Vmax×Cmax}</code> (Long), plus <code>masks ∈ {0,1}^{B×Vmax×Cmax}</code> (Bool).</li>
      <li>Create time-reversed versions <code>rev_x</code> and <code>rev_masks</code> by flipping only the <em>true</em> visits.</li>
      <li>Return <code>(x, masks, rev_x, rev_masks, y)</code> with <code>y ∈ ℝ^{B}</code> (Float).</li>
    </ul>
  </li>
  <li><strong>Split &amp; loaders</strong>: 80/20 split via <code>random_split</code>; <code>batch_size=32</code>; shuffle only the train loader.
      With the provided split, <code>len(train_loader)=25</code>.</li>
</ol>

<h3>2) Vectorized Masking Ops</h3>
<ul>
  <li><strong>sum_embeddings_with_mask</strong>: given <code>x_emb ∈ ℝ^{B×V×C×D}</code> and <code>masks ∈ {0,1}^{B×V×C}</code>,
      broadcast the mask and sum across codes → <code>ℝ^{B×V×D}</code> (no Python loops).</li>
  <li><strong>get_last_visit</strong>: from per-visit states <code>H ∈ ℝ^{B×V×D}</code> and <code>masks</code>,
      select the last <em>true</em> visit for each patient → <code>ℝ^{B×D}</code> (no loops; uses <code>any/sum/gather</code>).</li>
</ul>

<h3>3) Model — <code>NaiveRNN</code></h3>
<pre><code>Embedding(num_codes=619, emb_dim=128)
GRU_fwd(input_size=128, hidden_size=128, batch_first=True)
GRU_rev(input_size=128, hidden_size=128, batch_first=True)
Linear(256 → 1) + Sigmoid
</code></pre>
<p>
Forward pass: embed → <em>sum per visit (masked)</em> → GRU → <em>last-visit state</em>;
repeat on reversed stream; concatenate both last states (fwd + rev) → FC → Sigmoid → <code>p(y=1) ∈ (0,1)</code>.
</p>

<h3>4) Training &amp; Evaluation</h3>
<ul>
  <li><strong>Loss:</strong> <code>BCELoss</code> (model already outputs probabilities).</li>
  <li><strong>Optimizer:</strong> <code>Adam(lr=1e-3)</code>.</li>
  <li><strong>Metrics:</strong> Precision, Recall, F1 at threshold 0.5; ROC-AUC from probabilities.</li>
  <li><strong>Eval cadence:</strong> call <code>eval_model()</code> at the end of each epoch.</li>
</ul>

<hr/>

<h2>📈 Results (validation)</h2>
<ul>
  <li><strong>Before training</strong> (random init): acc/auc modest; baseline AUC ≈ 0.547.</li>
  <li><strong>After training (5 epochs)</strong>:
    <ul>
      <li>Epoch&nbsp;1 → AUC ≈ 0.820</li>
      <li>Epoch&nbsp;2 → AUC ≈ 0.833</li>
      <li>Epoch&nbsp;5 → <strong>AUC ≈ 0.832</strong>, Precision ≈ 0.741, Recall ≈ 0.777, F1 ≈ 0.758</li>
    </ul>
    The final model comfortably exceeds the assignment target of <strong>ROC-AUC &gt; 0.7</strong>.
  </li>
</ul>

<hr/>

<h2>🧪 Reproducibility</h2>
<ul>
  <li>Seeds set to <code>24</code> for Python, NumPy, and PyTorch; <code>PYTHONHASHSEED</code> set accordingly.</li>
  <li>Dtypes: <code>x, rev_x: long</code>; <code>masks, rev_masks: bool</code>; <code>y: float</code>.</li>
  <li>Batching invariants: consistent padding, visit-wise reversal (no flipping padded rows).</li>
</ul>

<hr/>

<h2>⚠️ Challenges &amp; How They Were Solved</h2>
<ol>
  <li>
    <strong>DataLoader signature</strong> — The notebook’s autograder calls
    <code>load_data(train_dataset, val_dataset, collate_fn)</code>. Implemented this signature
    exactly (no global paths), returning a shuffling train loader and a non-shuffling val loader.
  </li>
  <li>
    <strong>Masking without loops</strong> — Loops are disallowed/checked. Solved with pure tensor
    ops: broadcast the mask and <code>sum(dim=2)</code> for per-visit embeddings; compute last true
    visit via <code>any→sum→gather</code>.
  </li>
  <li>
    <strong>Reverse sequences correctly</strong> — Only flip the <em>real</em> visits, not padded rows;
    mirror the masks to match <code>rev_x</code>.
  </li>
  <li>
    <strong>Loss/optimizer timing</strong> — Define <code>BCELoss</code> and create the Adam optimizer
    on the instantiated model to avoid <code>NameError</code>.
  </li>
  <li>
    <strong>Metric consistency</strong> — Use probabilities for ROC-AUC, and 0.5-thresholded labels
    for precision/recall/F1; keep shapes as 1-D vectors.
  </li>
</ol>

<hr/>

<h2>🛠️ Environment</h2>
<pre><code>python &gt;= 3.9
pip install torch numpy scikit-learn
</code></pre>

<hr/>

<h2>🚀 How to Run</h2>
<ol>
  <li>Open <code>HW3_RNN.ipynb</code> and run cells top-to-bottom.</li>
  <li>Confirm the collate-function checks (dtypes and shapes) pass.</li>
  <li>Instantiate <code>NaiveRNN</code>, define loss/optimizer, and train for N epochs (e.g., 5).</li>
  <li>Verify final validation ROC-AUC &gt; 0.7.</li>
</ol>

<hr/>

<h2>📄 License</h2>
<p>MIT.</p>

<hr/>

<h2>🙌 Acknowledgements</h2>
<p>Thanks to the course staff for the dataset, scaffolding, and evaluation harness.</p>
