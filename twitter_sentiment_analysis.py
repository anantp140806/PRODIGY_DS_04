"""
Twitter Entity Sentiment Analysis
==================================
Analyzes and visualizes sentiment patterns across 30 entities
(Tech, Gaming, Health/Finance, Celebrities) from Twitter data.

Dataset : https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
Outputs : 4 separate PNG figures saved to the working directory

Requirements:
    pip install pandas numpy matplotlib scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
POS   = '#22C55E'
NEG   = '#EF4444'
NEU   = '#94A3B8'
IRR   = '#F59E0B'
BG    = '#0F1117'
CARD  = '#1A1D27'
CARD2 = '#22263A'
TEXT  = '#E2E8F0'
TSUB  = '#64748B'

plt.rcParams.update({
    'figure.facecolor' : BG,
    'axes.facecolor'   : CARD,
    'text.color'       : TEXT,
    'axes.labelcolor'  : TEXT,
    'xtick.color'      : TSUB,
    'ytick.color'      : TSUB,
    'axes.edgecolor'   : '#2D3148',
    'axes.grid'        : False,
    'font.family'      : 'DejaVu Sans',
})

# ── Step 1: Load or Generate Dataset ──────────────────────────────────────────
def load_or_generate_data(csv_path='twitter_entity_sentiment.csv'):
    """
    Loads dataset from csv_path if it exists.
    If not found, generates a synthetic dataset matching the Kaggle schema:
      columns: entity, sentiment, text
      sentiments: Positive, Negative, Neutral, Irrelevant
    """
    if os.path.exists(csv_path):
        print(f"Loading dataset from '{csv_path}' ...")
        df = pd.read_csv(csv_path)
        # Normalize column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]
        # Map common Kaggle column variations
        if 'tweet_content' in df.columns and 'text' not in df.columns:
            df = df.rename(columns={'tweet_content': 'text'})
        if 'entity' not in df.columns:
            possible = [c for c in df.columns if 'entity' in c or 'brand' in c or 'topic' in c]
            if possible:
                df = df.rename(columns={possible[0]: 'entity'})
        if 'sentiment' not in df.columns:
            possible = [c for c in df.columns if 'sentiment' in c or 'label' in c]
            if possible:
                df = df.rename(columns={possible[0]: 'sentiment'})
        print(f"  Loaded {len(df):,} rows.")
    else:
        print(f"'{csv_path}' not found — generating synthetic dataset ...")
        df = _generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        print(f"  Saved synthetic data to '{csv_path}'.")
    return df


def _generate_synthetic_data(n=71846, seed=42):
    np.random.seed(seed)

    entities = [
        'Microsoft', 'Apple', 'Google', 'Amazon', 'Facebook',
        'Twitter', 'Tesla', 'Netflix', 'Sony', 'Nintendo',
        'Borderlands', 'GTA', 'FIFA', 'CallOfDuty', 'Fortnite',
        'NBA2K', 'MaddenNFL', 'Cyberpunk2077', 'AssassinsCreed', 'Minecraft',
        'Johnson&Johnson', 'Pfizer', 'WorldHealthOrganization',
        'Bitcoin', 'Ethereum', 'Dogecoin',
        'TomHanks', 'LadyGaga', 'Cristiano', 'LeBronJames',
    ]
    sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    sent_weights = {
        'Microsoft'             : [0.40, 0.25, 0.25, 0.10],
        'Apple'                 : [0.42, 0.30, 0.18, 0.10],
        'Google'                : [0.38, 0.28, 0.24, 0.10],
        'Amazon'                : [0.35, 0.35, 0.20, 0.10],
        'Facebook'              : [0.22, 0.48, 0.20, 0.10],
        'Twitter'               : [0.28, 0.42, 0.20, 0.10],
        'Tesla'                 : [0.45, 0.25, 0.20, 0.10],
        'Netflix'               : [0.50, 0.22, 0.18, 0.10],
        'Sony'                  : [0.44, 0.22, 0.24, 0.10],
        'Nintendo'              : [0.55, 0.15, 0.20, 0.10],
        'Borderlands'           : [0.50, 0.20, 0.20, 0.10],
        'GTA'                   : [0.55, 0.18, 0.17, 0.10],
        'FIFA'                  : [0.35, 0.38, 0.17, 0.10],
        'CallOfDuty'            : [0.40, 0.32, 0.18, 0.10],
        'Fortnite'              : [0.42, 0.30, 0.18, 0.10],
        'NBA2K'                 : [0.38, 0.34, 0.18, 0.10],
        'MaddenNFL'             : [0.32, 0.40, 0.18, 0.10],
        'Cyberpunk2077'         : [0.30, 0.45, 0.15, 0.10],
        'AssassinsCreed'        : [0.48, 0.22, 0.20, 0.10],
        'Minecraft'             : [0.60, 0.12, 0.18, 0.10],
        'Johnson&Johnson'       : [0.30, 0.38, 0.22, 0.10],
        'Pfizer'                : [0.28, 0.42, 0.20, 0.10],
        'WorldHealthOrganization': [0.25, 0.45, 0.20, 0.10],
        'Bitcoin'               : [0.45, 0.28, 0.17, 0.10],
        'Ethereum'              : [0.43, 0.28, 0.19, 0.10],
        'Dogecoin'              : [0.48, 0.25, 0.17, 0.10],
        'TomHanks'              : [0.65, 0.10, 0.15, 0.10],
        'LadyGaga'              : [0.60, 0.14, 0.16, 0.10],
        'Cristiano'             : [0.58, 0.18, 0.14, 0.10],
        'LeBronJames'           : [0.50, 0.25, 0.15, 0.10],
    }
    tweet_templates = {
        'Positive'   : [
            "Love {e}! Absolutely amazing 🙌",
            "{e} never disappoints. Best ever!",
            "Just tried {e} — totally incredible. Recommend!",
            "{e} is killing it right now 🔥",
            "Shoutout to {e} for the amazing service!",
        ],
        'Negative'   : [
            "{e} is really going downhill lately smh",
            "Terrible experience with {e}. Never again.",
            "{e} needs to do better. This is unacceptable.",
            "Why does {e} keep making bad decisions?? 😤",
            "So disappointed in {e} after all these years.",
        ],
        'Neutral'    : [
            "{e} released a new update today",
            "Just saw the latest news about {e}",
            "Reading about {e} and their new policy",
            "Interesting article about {e} in the news",
        ],
        'Irrelevant' : [
            "{e} reminds me of something completely different",
            "Random thought: {e}",
            "I was thinking about {e} for some reason",
        ],
    }

    entity_col, sent_col, text_col = [], [], []
    per_entity = max(600, n // len(entities))
    for entity in entities:
        w = sent_weights[entity]
        sents = np.random.choice(sentiments, size=per_entity, p=w)
        for s in sents:
            entity_col.append(entity)
            sent_col.append(s)
            text_col.append(np.random.choice(tweet_templates[s]).replace('{e}', entity))

    df = pd.DataFrame({'entity': entity_col, 'sentiment': sent_col, 'text': text_col})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Step 2: Compute Analytics ──────────────────────────────────────────────────
def compute_analytics(df):
    print("\nComputing analytics ...")
    sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

    overall  = df['sentiment'].value_counts().to_dict()
    total    = len(df)

    # Top 15 entities by non-Irrelevant volume
    top15 = (df[df['sentiment'] != 'Irrelevant']['entity']
             .value_counts().head(15).index.tolist())

    # Sentiment breakdown per entity (%)
    sent_by_entity = {}
    for e in df['entity'].unique():
        sub   = df[df['entity'] == e]['sentiment'].value_counts()
        n_sub = sub.sum()
        sent_by_entity[e] = {s: round(sub.get(s, 0) / n_sub * 100, 1) for s in sentiments}

    # Net sentiment score
    sentiment_score = {}
    for e in df['entity'].unique():
        sub = df[df['entity'] == e]['sentiment'].value_counts()
        p   = sub.get('Positive', 0)
        n   = sub.get('Negative', 0)
        neu = sub.get('Neutral', 0)
        denom = p + n + neu
        sentiment_score[e] = round((p - n) / denom * 100, 1) if denom > 0 else 0

    ss_sorted = sorted(sentiment_score.items(), key=lambda x: x[1], reverse=True)

    # Category mapping
    categories = {
        'Tech'           : ['Microsoft','Apple','Google','Amazon','Facebook','Twitter','Tesla','Netflix'],
        'Gaming'         : ['Borderlands','GTA','FIFA','CallOfDuty','Fortnite','NBA2K','MaddenNFL',
                            'Cyberpunk2077','AssassinsCreed','Minecraft'],
        'Health/Finance' : ['Johnson&Johnson','Pfizer','WorldHealthOrganization',
                            'Bitcoin','Ethereum','Dogecoin'],
        'Celebrities'    : ['TomHanks','LadyGaga','Cristiano','LeBronJames','Sony','Nintendo'],
    }
    cat_sentiment = {}
    for cat, ents in categories.items():
        sub = df[df['entity'].isin(ents)]
        vc  = sub['sentiment'].value_counts()
        t   = vc.sum()
        cat_sentiment[cat] = {s: round(vc.get(s, 0) / t * 100, 1) for s in sentiments}

    top_pos = [(e, v) for e, v in ss_sorted if v > 0][:8]
    top_neg = [(e, v) for e, v in reversed(ss_sorted) if v < 0][:8]

    print(f"  Total tweets     : {total:,}")
    print(f"  Entities found   : {df['entity'].nunique()}")
    print(f"  Positivity rate  : {round(overall.get('Positive',0)/total*100,1)}%")
    print(f"  Most loved       : {top_pos[0][0]} (+{top_pos[0][1]}%)" if top_pos else "")
    print(f"  Most criticised  : {top_neg[0][0]} ({top_neg[0][1]}%)" if top_neg else "")

    return {
        'overall'         : overall,
        'total'           : total,
        'top15'           : top15,
        'sent_by_entity'  : sent_by_entity,
        'sentiment_score' : sentiment_score,
        'ss_sorted'       : ss_sorted,
        'cat_sentiment'   : cat_sentiment,
        'top_pos'         : top_pos,
        'top_neg'         : top_neg,
    }


# ── Step 3: Plot Functions ─────────────────────────────────────────────────────
def plot_fig1_overview(a, save_path='fig1_overview.png'):
    """KPI strip + Overall Donut + Category Grouped Bar"""
    overall = a['overall']; total = a['total']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.suptitle('Twitter Sentiment — Overview', fontsize=16,
                 fontweight='bold', color=TEXT, y=1.01)

    # KPI strip
    kpis = [
        (f"{total:,}",                                    'Total Tweets',    TEXT),
        (f"{overall.get('Positive', 0):,}",               'Positive',        POS),
        (f"{overall.get('Negative', 0):,}",               'Negative',        NEG),
        (f"{overall.get('Neutral', 0):,}",                'Neutral',         NEU),
        (f"{overall.get('Irrelevant', 0):,}",             'Irrelevant',      IRR),
        (f"{round(overall.get('Positive',0)/total*100,1)}%", 'Positivity Rate', POS),
    ]
    for i, (val, lbl, col) in enumerate(kpis):
        x = 0.08 + i * 0.155
        rect = FancyBboxPatch((x - 0.065, 0.88), 0.12, 0.11,
                              boxstyle='round,pad=0.01', linewidth=0,
                              facecolor=CARD2, transform=fig.transFigure, zorder=2)
        fig.add_artist(rect)
        fig.text(x, 0.96, val,  ha='center', fontsize=15, fontweight='bold',
                 color=col, transform=fig.transFigure)
        fig.text(x, 0.90, lbl,  ha='center', fontsize=8,
                 color=TSUB, transform=fig.transFigure)

    # Donut
    ax1 = axes[0]; ax1.set_facecolor(CARD)
    sent_labels = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    sent_colors = [POS, NEG, NEU, IRR]
    _, _, autotexts = ax1.pie(
        [overall.get(s, 0) for s in sent_labels],
        colors=sent_colors, startangle=90, autopct='%1.1f%%',
        pctdistance=0.75,
        wedgeprops={'width': 0.52, 'edgecolor': CARD, 'linewidth': 2.5},
        textprops={'color': TEXT, 'fontsize': 9, 'fontweight': 'bold'},
    )
    for at, c in zip(autotexts, sent_colors):
        at.set_color(c); at.set_fontsize(9)
    ax1.text(0, 0, f"{round(overall.get('Positive',0)/total*100,1)}%\nPositive",
             ha='center', va='center', fontsize=14, fontweight='bold', color=POS)
    ax1.set_title('Overall Sentiment Distribution', fontsize=12,
                  fontweight='bold', color=TEXT, pad=12)
    ax1.legend(handles=[mpatches.Patch(color=c, label=l)
                        for l, c in zip(sent_labels, sent_colors)],
               loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=2, fontsize=9, frameon=False)

    # Category grouped bar
    ax2 = axes[1]; ax2.set_facecolor(CARD)
    cats = list(a['cat_sentiment'].keys())
    x2   = np.arange(len(cats)); w = 0.22
    for i, (s, col) in enumerate(zip(['Positive', 'Negative', 'Neutral'], [POS, NEG, NEU])):
        vals = [a['cat_sentiment'][c][s] for c in cats]
        bars = ax2.bar(x2 + i * w - w, vals, w * 0.88, color=col, alpha=0.88, label=s, zorder=3)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                     f'{v:.0f}%', ha='center', va='bottom',
                     fontsize=9, color=col, fontweight='bold')
    ax2.set_xticks(x2); ax2.set_xticklabels(cats, fontsize=11, color=TEXT)
    ax2.set_ylabel('Share of Tweets (%)', fontsize=9, color=TSUB)
    ax2.set_ylim(0, 75)
    ax2.set_title('Sentiment by Category', fontsize=12,
                  fontweight='bold', color=TEXT, pad=12)
    ax2.legend(fontsize=9, frameon=False, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_fig2_net_score(a, save_path='fig2_net_score.png'):
    """Full horizontal bar chart — net sentiment for all entities"""
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG)
    ax.set_facecolor(CARD)
    fig.suptitle('Net Sentiment Score — All Entities\n(Positive − Negative) / Total × 100',
                 fontsize=14, fontweight='bold', color=TEXT, y=1.01)

    ss_sorted = a['ss_sorted']
    ents   = [e for e, _ in ss_sorted][::-1]   # highest at top
    scores = [v for _, v in ss_sorted][::-1]
    y      = np.arange(len(ents))

    bars = ax.barh(y, scores,
                   color=[POS if v >= 0 else NEG for v in scores],
                   alpha=0.85, height=0.65, zorder=3)
    for b, v in zip(bars, scores):
        xpos = v + 0.5 if v >= 0 else v - 0.5
        ax.text(xpos, b.get_y() + b.get_height() / 2,
                f'{v:+.1f}%', va='center', fontsize=9,
                color=(POS if v >= 0 else NEG), fontweight='bold',
                ha='left' if v >= 0 else 'right')

    ax.axvline(0, color='#3D4168', linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels(ents, fontsize=10)
    ax.set_xlabel('Net Sentiment Score (%)', fontsize=10, color=TSUB)
    ax.set_xlim(-50, 90)
    ax.legend(handles=[
        mpatches.Patch(color=POS, label='Net Positive'),
        mpatches.Patch(color=NEG, label='Net Negative'),
    ], fontsize=10, frameon=False, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_fig3_stacked(a, save_path='fig3_stacked.png'):
    """Stacked bar chart — top 15 entities"""
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    ax.set_facecolor(CARD)
    fig.suptitle('Sentiment Distribution — Top 15 Entities (Stacked)',
                 fontsize=14, fontweight='bold', color=TEXT)

    top15 = a['top15'][:15]
    sbe   = a['sent_by_entity']
    pv    = [sbe[e]['Positive']   for e in top15]
    nv    = [sbe[e]['Negative']   for e in top15]
    nev   = [sbe[e]['Neutral']    for e in top15]
    iv    = [100 - pv[i] - nv[i] - nev[i] for i in range(len(top15))]
    x4    = np.arange(len(top15))

    ax.bar(x4, pv, color=POS, alpha=0.9, label='Positive', zorder=3)
    ax.bar(x4, nv, bottom=pv, color=NEG, alpha=0.9, label='Negative', zorder=3)
    ax.bar(x4, nev, bottom=[pv[i]+nv[i] for i in range(len(top15))],
           color=NEU, alpha=0.9, label='Neutral', zorder=3)
    ax.bar(x4, iv, bottom=[pv[i]+nv[i]+nev[i] for i in range(len(top15))],
           color=IRR, alpha=0.9, label='Irrelevant', zorder=3)

    for xi, v in zip(x4, pv):
        ax.text(xi, v / 2, f'{v:.0f}%', ha='center', va='center',
                fontsize=8.5, color='white', fontweight='bold')

    ax.set_xticks(x4)
    ax.set_xticklabels(top15, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Sentiment Share (%)', fontsize=10, color=TSUB)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, frameon=False, loc='upper right', ncol=4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_fig4_extremes(a, save_path='fig4_extremes.png'):
    """Side-by-side most loved vs most criticised"""
    fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle('Brand Sentiment Extremes', fontsize=14,
                 fontweight='bold', color=TEXT)

    # Most loved
    ax5.set_facecolor(CARD)
    tp  = a['top_pos'][:8]
    pe  = [e for e, _ in tp]; ps = [v for _, v in tp]
    y5  = np.arange(len(pe))
    b5  = ax5.barh(y5, ps, color=POS, alpha=0.85, height=0.6, zorder=3)
    for b, v in zip(b5, ps):
        ax5.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
                 f'+{v}%', va='center', fontsize=10, color=POS, fontweight='bold')
    ax5.set_yticks(y5); ax5.set_yticklabels(pe, fontsize=11)
    ax5.set_xlim(0, max(ps) * 1.28)
    ax5.set_xlabel('Net Score (%)', fontsize=9, color=TSUB)
    ax5.set_title('Most Loved  ✓', fontsize=13, fontweight='bold', color=POS, pad=10)

    # Most criticised
    ax6.set_facecolor(CARD)
    tn  = a['top_neg'][:8]
    ne_e = [e for e, _ in tn]; ns = [abs(v) for _, v in tn]
    y6  = np.arange(len(ne_e))
    b6  = ax6.barh(y6, ns, color=NEG, alpha=0.85, height=0.6, zorder=3)
    for b, v, (_, raw) in zip(b6, ns, tn):
        ax6.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                 f'{raw}%', va='center', fontsize=10, color=NEG, fontweight='bold')
    ax6.set_yticks(y6); ax6.set_yticklabels(ne_e, fontsize=11)
    ax6.set_xlim(0, max(ns) * 1.28)
    ax6.set_xlabel('Neg Score (abs %)', fontsize=9, color=TSUB)
    ax6.set_title('Most Criticised  ✗', fontsize=13, fontweight='bold', color=NEG, pad=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved → {save_path}")


# ── Step 4: Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   Twitter Entity Sentiment Analysis")
    print("=" * 55)

    df        = load_or_generate_data('twitter_entity_sentiment.csv')
    analytics = compute_analytics(df)

    print("\nGenerating figures ...")
    plot_fig1_overview  (analytics, 'fig1_overview.png')
    plot_fig2_net_score (analytics, 'fig2_net_score.png')
    plot_fig3_stacked   (analytics, 'fig3_stacked.png')
    plot_fig4_extremes  (analytics, 'fig4_extremes.png')

    print("\nAll done! 4 PNG files saved to the current directory.")
    print("  fig1_overview.png   — KPI strip, donut, category bars")
    print("  fig2_net_score.png  — Net sentiment score (all entities)")
    print("  fig3_stacked.png    — Stacked distribution (top 15)")
    print("  fig4_extremes.png   — Most loved vs most criticised")


if __name__ == '__main__':
    main()
