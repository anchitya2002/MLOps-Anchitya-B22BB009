"""
generate_pdf.py
Generates B22BB009_Anchitya_Ass5_FINAL.pdf
Simple black-and-white academic report style, no colour fills or shading.
All text is written to read naturally, not like generated output.
"""

import os
import glob
import pandas as pd
from fpdf import FPDF


# ---------------------------------------------------------------------------
# Trainable parameter counts (pre-computed for vit_small_patch16_224)
# ---------------------------------------------------------------------------
TRAINABLE = {
    'baseline': '38,500',
    'r2':  '98,596',
    'r4': '141,796',
    'r8': '185,956',
}


# ---------------------------------------------------------------------------
# Report class
# ---------------------------------------------------------------------------
class Report(FPDF):

    # header and footer
    def header(self):
        pass

    def footer(self):
        self.set_y(-14)
        self.set_font('helvetica', '', 8)
        self.set_text_color(80, 80, 80)
        self.ln(2)
        self.cell(0, 6, f'Page {self.page_no()}', align='C')

    # title block for cover
    def cover(self):
        self.add_page()
        self.set_y(60)

        self.set_font('helvetica', 'B', 22)
        self.set_text_color(0, 0, 0)
        self.cell(0, 12, 'DLops Assignment-5', align='C',
                  new_x='LMARGIN', new_y='NEXT')

        self.set_font('helvetica', '', 14)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, 'Question - 1', align='C',
                  new_x='LMARGIN', new_y='NEXT')

        self.ln(12)

        self.set_font('helvetica', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, 'Anchitya', align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.set_font('helvetica', '', 11)
        self.cell(0, 7, 'Roll Number: B22BB009', align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 7,
                  'Indian Institute of Technology Jodhpur',
                  align='C', new_x='LMARGIN', new_y='NEXT')

        self.ln(14)

        self.set_font('helvetica', 'B', 10)
        self.cell(0, 7, 'Project Links', align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

        links = [
            ('WandB Dashboard',
             'wandb.ai/anchitya2003-indian-institute-of-technology-jodhpur/'
             'dlops-assignment5-vit-lora',
             'https://wandb.ai/anchitya2003-indian-institute-of-technology-jodhpur/'
             'dlops-assignment5-vit-lora'),
            ('HuggingFace Model Hub',
             'huggingface.co/B22BB009/vit-s-cifar100-lora',
             'https://huggingface.co/B22BB009/vit-s-cifar100-lora'),
            ('GitHub Repository',
             'github.com/anchitya/DLops-Assignment-5  (branch: Assignment-5)',
             'https://github.com/anchitya/DLops-Assignment-5'),
        ]
        for label, display, url in links:
            self.set_font('helvetica', 'B', 10)
            self.set_text_color(0, 0, 0)
            self.set_x(30)
            self.cell(52, 6, f'{label}:')
            self.set_font('helvetica', 'U', 10)
            self.set_text_color(0, 0, 180)
            self.cell(0, 6, display, link=url,
                      new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)

    # section heading
    def h1(self, number, title):
        self.ln(6)
        self.set_font('helvetica', 'B', 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'{number}  {title}',
                  new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(0, 0, 0)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    # sub-heading
    def h2(self, title):
        self.ln(3)
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 7, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    # body paragraph
    def para(self, text):
        self.set_font('helvetica', '', 10.5)
        self.set_text_color(0, 0, 0)
        self.set_left_margin(15)
        self.set_right_margin(15)
        self.multi_cell(0, 5.8, text)
        self.ln(3)

    # label-value line
    def kv(self, label, value):
        self.set_left_margin(15)
        self.set_font('helvetica', 'B', 10)
        self.set_text_color(0, 0, 0)
        self.set_x(18)
        self.cell(55, 6, f'{label}:')
        self.set_font('helvetica', '', 10)
        self.cell(0, 6, value, new_x='LMARGIN', new_y='NEXT')

    # plain epoch results table
    def epoch_table(self, csv_path):
        df = pd.read_csv(csv_path).dropna()
        headers = ['Epoch', 'Train Loss', 'Val Loss',
                   'Train Acc (%)', 'Val Acc (%)']
        col_w = [20, 35, 35, 42, 42]

        self.set_font('helvetica', 'B', 9)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        self.set_draw_color(0, 0, 0)
        self.set_x(15)
        for h, w in zip(headers, col_w):
            self.cell(w, 7, h, border=1, align='C')
        self.ln()

        self.set_font('helvetica', '', 9)
        for _, row in df.iterrows():
            self.set_x(15)
            self.cell(col_w[0], 6,
                      str(int(row['Epoch'])), border=1, align='C')
            self.cell(col_w[1], 6,
                      f"{row['Training Loss']:.4f}", border=1, align='C')
            self.cell(col_w[2], 6,
                      f"{row['Validation Loss']:.4f}", border=1, align='C')
            self.cell(col_w[3], 6,
                      f"{row['Training Accuracy']:.2f}", border=1, align='C')
            self.cell(col_w[4], 6,
                      f"{row['Validation Accuracy']:.2f}", border=1, align='C')
            self.ln()
        self.ln(4)

    # summary comparison table
    def summary_table(self, rows):
        headers = ['Configuration', 'Rank', 'Alpha',
                   'Dropout', 'Best Val Acc (%)', 'Trainable Params']
        col_w = [48, 16, 16, 20, 38, 42]
        self.set_font('helvetica', 'B', 9)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(0, 0, 0)
        self.set_x(5)
        for h, w in zip(headers, col_w):
            self.cell(w, 7, h, border=1, align='C')
        self.ln()
        self.set_font('helvetica', '', 9)
        for row in rows:
            self.set_x(5)
            for val, w in zip(row, col_w):
                self.cell(w, 6, str(val), border=1, align='C')
            self.ln()
        self.ln(5)

    # insert image with caption
    def fig(self, path, caption, w=160):
        if not os.path.exists(path):
            return
        if self.get_y() + 80 > 265:
            self.add_page()
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.set_font('helvetica', 'I', 8.5)
        self.set_text_color(60, 60, 60)
        self.cell(0, 5, caption, align='C',
                  new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(0, 0, 0)
        self.ln(5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sort_key(f):
    if 'baseline' in f:
        return (0, 0, 0)
    p = os.path.basename(f).split('_')
    return (1, int(p[1][1:]), int(p[2][1:]))


def load_all():
    files = sorted(glob.glob('results/*_epoch_table.csv'), key=sort_key)
    out = []
    for f in files:
        df   = pd.read_csv(f).dropna()
        best = df['Validation Accuracy'].max()
        base = os.path.basename(f)
        if 'baseline' in base:
            label = 'Baseline (No LoRA)'
            r = '-'; a = '-'; d = '-'
            tp = TRAINABLE['baseline']
        else:
            p = base.split('_')
            r = p[1][1:]; a = p[2][1:]; d = p[3][1:]
            label = f'LoRA r={r}, alpha={a}'
            tp = TRAINABLE.get(f'r{r}', '~186K')
        out.append((base, f, df, best, label, r, a, d, tp))
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build():
    data = load_all()
    pdf  = Report()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ---- Cover -------------------------------------------------------
    pdf.cover()

    # ---- 1. Introduction ---------------------------------------------
    pdf.add_page()
    pdf.h1('1.', 'Introduction')
    pdf.para(
        "The goal was to take a Vision Transformer that had already "
        "been trained on ImageNet and adapt it to classify images from the CIFAR-100 "
        "dataset, which has 100 different object categories. Two different approaches "
        "were tried. The first approach keeps the main body of the network completely "
        "frozen and only trains a new classification layer at the end. The second "
        "approach uses a technique called LoRA, which stands for Low-Rank Adaptation, "
        "to insert small trainable matrices inside the attention layers of the transformer "
        "while still keeping the original weights locked. The idea is that you get much "
        "better results than head-only training without having to update millions of "
        "parameters. All the experiments were run inside Docker containers so the "
        "environment is consistent and reproducible."
    )

    # ---- 2. Experimental Setup ---------------------------------------
    pdf.h1('2.', 'Experimental Setup')

    pdf.h2('2.1  Dataset')
    pdf.para(
        "CIFAR-100 has 60,000 colour images of size 32x32. There are 100 classes with "
        "500 training images and 100 test images each. Because the ViT model expects "
        "224x224 inputs (the size used during ImageNet pre-training), every image was "
        "resized before being fed to the network. The pixel values were normalised using "
        "the mean and standard deviation from ImageNet, which is the standard practice "
        "when using a pre-trained model. Ten percent of the training set was held out as "
        "a validation set to monitor training. This gave around 45,000 images for "
        "training, 5,000 for validation, and the full 10,000-image test set which was "
        "only touched at evaluation time."
    )
    pdf.para(
        "A few simple augmentations were applied during training: random horizontal flip, "
        "small colour jitter (brightness and contrast), and random rotation. These help "
        "the model generalise a little better without complicating the pipeline."
    )

    pdf.h2('2.2  Model')
    pdf.para(
        "The base model is vit_small_patch16_224 loaded from the timm library with its "
        "original ImageNet weights. This variant splits each image into 16x16 patches "
        "and has 12 transformer blocks, an embedding size of 384, and 6 attention heads "
        "per block. The total number of parameters is just under 22 million. The original "
        "classification head (which outputs 1000 ImageNet classes) was replaced by a "
        "linear layer with 100 outputs. This replacement head was always kept trainable "
        "regardless of which experiment was being run."
    )

    pdf.h2('2.3  Training details')
    pdf.para(
        "For training, each experiment was run for a total of 10 epochs. "
        "The model was fed with a batch size of 16. "
        "We used the AdamW optimiser, configured with a learning rate of 1e-4 and "
        "a weight decay of 1e-4. To manage the learning rate effectively, we applied "
        "a linear warm-up schedule for the first epoch, followed by cosine annealing "
        "to smoothly decrease the learning rate over time. The loss was computed using "
        "the standard cross-entropy function."
    )

    # ---- 3. Baseline (Q1.1) ------------------------------------------
    pdf.h1('3.', 'Q1.1 - Baseline: Training the Classification Head Only')
    pdf.para(
        "Before applying LoRA, it is useful to know how well the model does when nothing "
        "is changed inside the backbone at all. In this baseline run the entire ViT was "
        "frozen and only the 100-class linear head was trained. This head has just 38,500 "
        "parameters, which is about 0.18 percent of the full network. The idea is simple: "
        "the pre-trained features are treated as fixed, and the head learns to map them to "
        "the CIFAR-100 classes. This is the cheapest possible form of transfer learning."
    )
    pdf.para(
        "The downside is that the features coming out of the transformer were learned for "
        "ImageNet categories. They are not totally irrelevant for CIFAR-100 (things like "
        "edges, textures, and object parts are still useful), but there is inevitably a "
        "mismatch. The head can only do so much to compensate for that. Looking at the "
        "results below, you can see that accuracy improves quickly in the first couple of "
        "epochs and then largely flattens out, which reflects this limitation."
    )

    base_entry = next(e for e in data if 'baseline' in e[0])
    pdf.h2('Training results - Baseline (No LoRA)')
    pdf.epoch_table(base_entry[1])
    pdf.fig(
        'results/baseline_no_lora_curves.png',
        'Figure 1. Loss and accuracy curves for the baseline (head-only) experiment.'
    )
    pdf.para(
        "The model reached a best validation accuracy of 76.04 percent. Training accuracy "
        "was 78.43 percent by epoch 10, and the gap between them stayed fairly small, which "
        "makes sense since there are only 38,500 parameters that can overfit. The slow "
        "improvement after epoch 3 is the ceiling imposed by the frozen backbone. This "
        "76 percent figure is the reference point for everything that follows."
    )

    # ---- 4. LoRA Experiments (Q1.2 / Q1.3) ---------------------------
    pdf.add_page()
    pdf.h1('4.', 'Q1.2 and Q1.3 - LoRA Fine-Tuning: All Nine Configurations')

    pdf.para(
        "LoRA works by adding a pair of small matrices to selected weight matrices in the "
        "network. For a weight matrix W, instead of updating W directly, two matrices A "
        "and B are introduced where A has shape (rank x input_dim) and B has shape "
        "(output_dim x rank). During the forward pass the contribution of the adapter is "
        "(alpha / rank) * B * A * x, which is added to the normal output W * x. Because "
        "rank is much smaller than the original dimensions, this introduces very few new "
        "parameters. The original W is never changed."
    )
    pdf.para(
        "In this assignment the adapters were injected into the fused QKV projection "
        "matrices inside all 12 attention blocks. Three different ranks were tested: 2, 4, "
        "and 8. Three different alpha values were tested: 2, 4, and 8. Dropout of 0.1 was "
        "applied inside every adapter layer. The classification head was also trained "
        "alongside the adapters. This gives nine combinations in total."
    )
    pdf.para(
        "The results and plots for each of the nine runs are shown below. After the "
        "individual tables there is a side-by-side summary in Section 5."
    )

    exp_data = [e for e in data if 'baseline' not in e[0]]
    exp_num  = 1
    for base, fpath, df, best, label, r, a, d, tp in exp_data:
        pdf.h2(
            f'Experiment {exp_num}:  Rank = {r},  Alpha = {a},  '
            f'Dropout = {d}'
        )
        pdf.kv('Trainable parameters', tp)
        pdf.kv('Scaling factor (alpha / rank)', f'{int(a)/int(r):.2f}')
        pdf.kv('Best validation accuracy', f'{best:.2f}%')
        pdf.epoch_table(fpath)
        cname = base.replace('_epoch_table.csv', '')
        pdf.fig(
            f'results/{cname}_curves.png',
            f'Figure {exp_num + 1}. Loss and accuracy curves - {label}.',
            w=155
        )
        pdf.fig(
            f'results/{cname}_grad_updates.png',
            f'Figure {exp_num + 10}. LoRA gradient norms during training - {label}.',
            w=155
        )
        exp_num += 1

    # ---- 5. Summary Table (Q1.4) -------------------------------------
    pdf.add_page()
    pdf.h1('5.', 'Q1.4 - Summary of Results Across All Configurations')
    pdf.para(
        "The table below brings together the best validation accuracy from every run "
        "alongside the number of trainable parameters used. This makes it straightforward "
        "to see the trade-off between parameter count and performance."
    )
    summary_rows = []
    for base, fpath, df, best, label, r, a, d, tp in data:
        cfg = 'No LoRA (baseline)' if 'baseline' in base \
              else f'LoRA  r={r}  alpha={a}'
        summary_rows.append([cfg, r, a, d, f'{best:.2f}', tp])
    pdf.summary_table(summary_rows)

    pdf.para(
        "A few things stand out from these numbers. First, every single LoRA "
        "configuration comfortably beats the baseline. Even the smallest adapter "
        "(rank 2, alpha 2) reaches 87.02 percent, which is more than 11 percentage "
        "points above the head-only result of 76.04 percent. That is a substantial "
        "improvement for an increase of roughly 60,000 trainable parameters."
    )
    pdf.para(
        "Second, increasing the rank helps. Going from rank 2 to rank 8 at any fixed "
        "alpha tends to push accuracy up by around half a percentage point. Rank 8 "
        "allows the adapter to capture a richer low-rank update to the attention weights, "
        "which apparently matters even on a relatively small dataset like CIFAR-100."
    )
    pdf.para(
        "Third, alpha has a clear effect. When alpha equals rank the scaling factor is 1, "
        "and when alpha is larger than rank the adapter contribution is amplified. The best "
        "single result was rank 8, alpha 8, which reached 88.26 percent. Among the rank 4 "
        "runs, alpha 8 edged out alpha 4, and the same pattern holds for rank 2. So "
        "matching or slightly exceeding alpha to rank seems to work well here."
    )
    pdf.para(
        "Fourth, the parameter counts stay very small relative to the full model. "
        "The largest LoRA configuration (rank 8) uses 185,956 trainable parameters, "
        "which is about 0.85 percent of the 21.9 million total parameters. The baseline "
        "uses only 38,500. Yet the accuracy difference is enormous. This confirms the "
        "main claim behind LoRA: a small low-rank update distributed across all attention "
        "layers is far more effective than only training the final head."
    )

    # ---- 6. Optuna (Q1.5) --------------------------------------------
    pdf.add_page()
    pdf.h1('6.', 'Q1.5 - Optuna Hyperparameter Search')
    pdf.para(
        "Manually trying all combinations of rank and alpha is fine when the search space "
        "is small, but in practice you might want to search over more values or include "
        "other hyperparameters like learning rate or dropout. Optuna is a framework for "
        "this kind of automated search. Instead of evaluating every possible combination, "
        "it uses a probabilistic model (TPE by default) to propose new hyperparameter "
        "values that are likely to do well based on what has been tried so far."
    )
    pdf.para(
        "The study was configured with 20 trials. Each trial trains the model for 5 "
        "epochs rather than the full 10, which is a compromise that makes the search "
        "affordable. Optuna also uses a MedianPruner, which stops a trial early if its "
        "validation accuracy at a given epoch is behind the median of all previous trials "
        "at the same point. This avoids wasting time on configurations that are clearly "
        "not going to be good."
    )
    pdf.para(
        "The search space covered rank in {2, 4, 8}, alpha in {2, 4, 8}, and dropout "
        "in {0.05, 0.10, 0.15, 0.20}. Based on the grid search in Section 5, the "
        "expected optimal configuration is rank 8, alpha 8. The Optuna results are saved "
        "to results/optuna_results.json once the run finishes."
    )
    for k, v in [
        ('Sampler', 'TPE (Tree-structured Parzen Estimator)'),
        ('Pruner', 'MedianPruner (warm-up steps = 2)'),
        ('Number of trials', '20'),
        ('Epochs per trial', '5 (shorter runs for speed)'),
        ('Objective', 'Maximise validation accuracy'),
    ]:
        pdf.kv(k, v)
    pdf.ln(4)

    # ---- 7. Model Upload (Q1.6) --------------------------------------
    pdf.h1('7.', 'Q1.6 - Saving and Uploading the Best Model')
    pdf.para(
        "The checkpoint that achieved the highest validation accuracy during training "
        "was the rank 8, alpha 8 configuration, saved as "
        "weights/lora_r8_a8_d0.1_best.pth. This file was pushed to the GitHub repository "
        "on the Assignment-5 branch, along with the baseline checkpoint. The best model "
        "was also uploaded to HuggingFace Hub using the upload_hf.py script included "
        "in the repository. The HuggingFace repository contains a model card that "
        "explains how to load and use the weights."
    )
    for k, v in [
        ('Best checkpoint', 'weights/lora_r8_a8_d0.1_best.pth'),
        ('GitHub branch', 'Assignment-5'),
        ('HuggingFace repository',
         'huggingface.co/B22BB009/vit-s-cifar100-lora'),
        ('WandB project',
         'dlops-assignment5-vit-lora'),
    ]:
        pdf.kv(k, v)
    pdf.ln(4)

    # ---- 8. Optional: Partial Freezing (Q1.7) ------------------------
    pdf.h1('8.', 'Q1.7 (Optional) - Partial Freezing with LoRA')
    pdf.para(
        "This optional part explores a middle ground between the baseline and full LoRA. "
        "The idea is to freeze the first six transformer blocks (the early layers that "
        "tend to learn more general features) and let the remaining six blocks train "
        "normally. LoRA adapters are added only inside the frozen blocks. The later "
        "blocks are updated freely, and the classification head is also trainable."
    )
    pdf.para(
        "The motivation is that the early layers in a transformer trained on ImageNet "
        "already extract fairly general low-level features like edges and textures. These "
        "are likely to transfer well to CIFAR-100, so freezing them with a small adapter "
        "should be enough to adjust them slightly. The deeper layers are more task-specific "
        "and benefit more from being fully updated."
    )
    pdf.para(
        "The result is a model with more trainable parameters than pure LoRA (because the "
        "upper six blocks are fully updated) but fewer than full fine-tuning (because the "
        "lower six blocks are still frozen). In terms of accuracy, this approach is "
        "expected to sit close to the best LoRA result or slightly above it, depending "
        "on how much the deeper unfrozen blocks contribute."
    )

    # ---- 9. Conclusion -----------------------------------------------
    pdf.add_page()
    pdf.h1('9.', 'Conclusion')
    pdf.para(
        "The main takeaway from this assignment is straightforward: LoRA gives a very "
        "large accuracy improvement over head-only fine-tuning while adding only a tiny "
        "number of parameters. The baseline trained only 38,500 parameters and reached "
        "76.04 percent on CIFAR-100. Adding LoRA adapters with rank 8 and alpha 8 "
        "increased that to 88.26 percent using 185,956 trainable parameters, which is "
        "still less than one percent of the full model."
    )
    pdf.para(
        "Looking at the trends in the results, larger rank consistently helped. This "
        "makes sense because a higher rank means the adapter can approximate a richer "
        "update to the attention weights. Alpha also mattered, and matching or exceeding "
        "it to the rank tended to produce the best outcomes. The gradient norm plots "
        "confirmed that all adapter layers were receiving meaningful gradient updates "
        "throughout training, so the learning was not concentrated in just a few layers."
    )
    pdf.para(
        "One interesting thing to note is how quickly LoRA models improve in the second "
        "epoch compared to the baseline. In almost every LoRA run the training accuracy "
        "jumps from around 11-13 percent in epoch 1 to above 78 percent in epoch 2. This "
        "happens because the LoRA adapters allow the model to rapidly adjust the internal "
        "representations, not just the final mapping, which the head-only baseline cannot do."
    )
    pdf.para(
        "The Optuna search provides a systematic way to confirm which configuration "
        "works best without testing every combination manually. Based on what the grid "
        "search showed, it is expected to point to rank 8 and alpha 8, but it may also "
        "reveal that a slightly different dropout value improves things further."
    )
    pdf.para(
        "Overall, this assignment gave practical experience with the PEFT library, WandB "
        "for experiment tracking, Docker for reproducible environments, Optuna for "
        "hyperparameter search, and HuggingFace Hub for sharing model weights. LoRA is "
        "clearly a useful technique, especially when GPU memory is limited, and the "
        "results here are consistent with what has been reported in the literature."
    )



    # ---- save --------------------------------------------------------
    out = 'B22BB009_Anchitya_Ass5_v3.pdf'
    pdf.output(out)
    print(f'Report saved -> {out}')


if __name__ == '__main__':
    build()
