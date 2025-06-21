#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path
import matplotlib.patheffects as PathEffects

# Set the figure size and style - larger to accommodate all elements
plt.figure(figsize=(24, 18))
plt.style.use('seaborn-v0_8-whitegrid')

# Define colors
colors = {
    # Figure 1 genes (LPS modification) - Right quarter
    'ushA': '#3498db',  # Blue
    'rfaL': '#e74c3c',  # Red
    'rhmD': '#2ecc71',  # Green
    
    # Figure 2 genes (Biofilm formation) - Left quarter
    'lsrR': '#9b59b6',  # Purple
    'bdcA': '#f39c12',  # Orange
    'ratB': '#16a085',  # Teal
    
    # Figure 3 genes (Metabolic and Regulatory) - Middle half
    'leuD_2': '#a1d344',  # Yellow-green
    'idnK': '#a1d344',    # Yellow-green
    'frdD': '#a1d344',    # Yellow-green
    'fepG': '#d35400',    # Brown
    'punR': '#e84393',    # Magenta
    'rlmD': '#00cec9',    # Cyan
    'leuS': '#00cec9',    # Cyan
    'casA': '#6c5ce7',    # Dark purple
    
    # Cell components - Updated colors as requested
    'outer_membrane': '#d2b48c',  # Light brown
    'inner_membrane': '#add8e6',  # Light blue
    'cytoplasm': '#ecf0f1',       # Very light gray
    
    # Other elements
    'biofilm': '#e67e22',      # Dark orange
    'bacteria': '#c0392b',     # Dark red
    'colistin': '#8e44ad',     # Dark purple
    'lps': '#f1c40f',          # Yellow
    'metabolic': '#a1d344',    # Yellow-green
    'regulatory': '#e84393',   # Magenta
    'translation': '#00cec9',  # Cyan
    'transport': '#d35400',    # Brown
    'genome': '#6c5ce7',       # Dark purple
    'arrow': '#7f8c8d',        # Gray
    'background': '#f9f9f9',   # Light gray
    'text': '#2c3e50'          # Dark blue
}

# Create a background
ax = plt.gca()
ax.set_facecolor(colors['background'])
ax.grid(False)

# Define abbreviations for mechanisms
mechanism_abbr = {
    "Quorum sensing": "QS",
    "c-di-GMP signaling": "cGS",
    "Surface properties": "SP",
    "Enhanced biofilm formation": "EBF",
    "Physical barrier to colistin": "PBC",
    "Leucine biosynthesis": "LB",
    "Carbon metabolism": "CM",
    "Anaerobic respiration": "AR",
    "Iron acquisition": "IA",
    "Purine metabolism": "PM",
    "Ribosome modification": "RM",
    "Protein synthesis": "PS",
    "Genome plasticity": "GP",
    "Altered membrane composition": "AMC",
    "Modified membrane permeability": "MMP",
    "UDP-sugar metabolism": "USM",
    "O-antigen attachment": "OAA",
    "Rhamnose metabolism": "RhM",
    "Modified LPS structure": "MLS",
    "Reduced colistin binding": "RCB"
}

# Store gene descriptions for right side column
genes_description = {
    # Biofilm formation genes
    'lsrR': "LuxS-regulated repressor",
    'bdcA': "Biofilm dispersal protein",
    'ratB': "Secreted protein",
    
    # Metabolic genes
    'leuD_2': "3-isopropylmalate dehydratase",
    'idnK': "Gluconate kinase",
    'frdD': "Fumarate reductase",
    'fepG': "Ferric enterobactin transport",
    
    # Regulatory genes
    'punR': "Purine nucleotide repressor",
    'rlmD': "23S rRNA methyltransferase",
    'leuS': "Leucyl-tRNA synthetase",
    'casA': "CRISPR-associated protein",
    
    # LPS modification genes
    'ushA': "UDP-sugar hydrolase",
    'rfaL': "O-antigen ligase",
    'rhmD': "L-rhamnonate dehydratase"
}

# Function to create an arrow with text
def create_arrow_with_text(start_x, start_y, end_x, end_y, text, color=colors['arrow'], 
                          text_offset_x=0, text_offset_y=0.02, connectionstyle="arc3,rad=0.1",
                          position="above"):
    # Create arrow
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        connectionstyle=connectionstyle, 
        arrowstyle="Simple,head_width=8,head_length=8",
        color=color, linewidth=2
    )
    ax.add_patch(arrow)
    
    # Only add text if provided
    if text:
        # Calculate midpoint for text placement
        mid_x = (start_x + end_x) / 2 + text_offset_x
        mid_y = (start_y + end_y) / 2 + text_offset_y
        
        # Use abbreviation if available
        abbr_text = mechanism_abbr.get(text, text)
        
        # Add text
        plt.text(mid_x, mid_y, abbr_text, 
                ha='center', va='center', 
                fontsize=10, fontweight='bold', color=colors['text'],
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Function to create a gene circle (without description)
def create_gene_circle(x, y, name, color, radius=0.025):
    gene_circle = patches.Circle(
        (x, y), radius, 
        facecolor=color, alpha=0.8, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(gene_circle)
    
    # Add gene name with larger font size and bold
    text = plt.text(x, y, name, 
                   ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
    text.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='black')])

# Function to display gene descriptions on the right side
def display_genes_description():
    # Set up column positions - adjusted for new cell position
    right_x = 0.90
    start_y = 0.75
    line_height = 0.03
    
    # Add column header
    plt.text(right_x, start_y + line_height, "Genes Description", 
             ha='center', va='center', 
             fontsize=24, fontweight='bold', color=colors['text'])
    
    # Add section headers and genes
    # Biofilm Formation Genes
    current_y = start_y
    plt.text(right_x, current_y, "Biofilm Formation Genes", 
             ha='center', va='center', 
             fontsize=20, fontweight='bold', color=colors['text'])
    
    current_y -= line_height
    for gene in ['lsrR', 'bdcA', 'ratB']:
        # Create small colored circle for gene
        gene_marker = patches.Circle(
            (right_x - 0.08, current_y), 0.008, 
            facecolor=colors[gene], alpha=0.8, edgecolor='black', linewidth=1
        )
        ax.add_patch(gene_marker)
        
        # Add gene name and description - now bold
        plt.text(right_x - 0.06, current_y, f"{gene}: {genes_description[gene]}", 
                 ha='left', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
        current_y -= line_height
    
    # Metabolic and Regulatory Genes
    current_y -= line_height/2
    plt.text(right_x, current_y, "Metabolic and Regulatory Genes", 
             ha='center', va='center', 
             fontsize=20, fontweight='bold', color=colors['text'])
    
    current_y -= line_height
    for gene in ['leuD_2', 'idnK', 'frdD', 'fepG', 'punR', 'rlmD', 'leuS', 'casA']:
        # Create small colored circle for gene
        gene_marker = patches.Circle(
            (right_x - 0.08, current_y), 0.008, 
            facecolor=colors[gene], alpha=0.8, edgecolor='black', linewidth=1
        )
        ax.add_patch(gene_marker)
        
        # Add gene name and description - now bold
        plt.text(right_x - 0.06, current_y, f"{gene}: {genes_description[gene]}", 
                 ha='left', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
        current_y -= line_height
    
    # LPS Modification Genes
    current_y -= line_height/2
    plt.text(right_x, current_y, "LPS Modification Genes", 
             ha='center', va='center', 
             fontsize=20, fontweight='bold', color=colors['text'])
    
    current_y -= line_height
    for gene in ['ushA', 'rfaL', 'rhmD']:
        # Create small colored circle for gene
        gene_marker = patches.Circle(
            (right_x - 0.08, current_y), 0.008, 
            facecolor=colors[gene], alpha=0.8, edgecolor='black', linewidth=1
        )
        ax.add_patch(gene_marker)
        
        # Add gene name and description - now bold
        plt.text(right_x - 0.06, current_y, f"{gene}: {genes_description[gene]}", 
                 ha='left', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
        current_y -= line_height

# Function to display mechanism abbreviations below colistin resistance
def display_mechanism_abbreviations_bottom():
    # Set up grid positions
    start_y = 0.15
    y_spacing = 0.02
    
       # Define abbreviations by pathway type
    biofilm_abbrs = [
        ('QS', 'Quorum sensing'), 
        ('cGS', 'c-di-GMP signaling'), 
        ('SP', 'Surface properties'), 
        ('EBF', 'Enhanced biofilm formation'),
        ('PBC', 'Physical barrier to colistin')
    ]
    
    metabolic_abbrs = [
        ('LB', 'Leucine biosynthesis'), 
        ('CM', 'Carbon metabolism'), 
        ('AR', 'Anaerobic respiration'), 
        ('IA', 'Iron acquisition'), 
        ('PM', 'Purine metabolism'),
        ('RM', 'Ribosome modification'), 
        ('PS', 'Protein synthesis'),
        ('MMP', 'Modified membrane permeability'), 
        ('AMC', 'Altered membrane composition'), 
        ('GP', 'Genome plasticity')
    ]
    
    lps_abbrs = [
        ('USM', 'UDP-sugar metabolism'), 
        ('OAA', 'O-antigen attachment'), 
        ('RhM', 'Rhamnose metabolism'), 
        ('MLS', 'Modified LPS structure'), 
        ('RCB', 'Reduced colistin binding')
    ]
    
    # Create bordered sections for each pathway type
    # Biofilm Formation Pathway
    biofilm_x = 0.2
    biofilm_width = 0.2
    biofilm_height = len(biofilm_abbrs) * y_spacing + 0.04  # Extra space for header
    
    # Create border for biofilm abbreviations
    biofilm_border = patches.FancyBboxPatch(
        (biofilm_x - biofilm_width/2, start_y - biofilm_height), 
        biofilm_width, biofilm_height,
        boxstyle=patches.BoxStyle("Round", pad=0.02),
        facecolor='white', alpha=0.8, edgecolor=colors['lsrR'], linewidth=2
    )
    ax.add_patch(biofilm_border)
    
    # Add header
    plt.text(biofilm_x, start_y - 0.02, "Biofilm Formation", 
             ha='center', va='center', 
             fontsize=16, fontweight='bold', color=colors['lsrR'])
    
    # Add abbreviations
    for i, (abbr, full) in enumerate(biofilm_abbrs):
        y_pos = start_y - 0.04 - i * y_spacing
        plt.text(biofilm_x, y_pos, f"{abbr}: {full}", 
                 ha='center', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
    
    # Metabolic and Regulatory Pathways
    metabolic_x = 0.5
    metabolic_width = 0.3
    metabolic_height = (len(metabolic_abbrs) // 2 + 1) * y_spacing + 0.04  # Extra space for header, 2 columns
    
    # Create border for metabolic abbreviations
    metabolic_border = patches.FancyBboxPatch(
        (metabolic_x - metabolic_width/2, start_y - metabolic_height), 
        metabolic_width, metabolic_height,
        boxstyle=patches.BoxStyle("Round", pad=0.02),
        facecolor='white', alpha=0.8, edgecolor=colors['metabolic'], linewidth=2
    )
    ax.add_patch(metabolic_border)
    
    # Add header
    plt.text(metabolic_x, start_y - 0.02, "Metabolic and Regulatory", 
             ha='center', va='center', 
             fontsize=16, fontweight='bold', color=colors['metabolic'])
    
    # Add abbreviations in 2 columns
    for i, (abbr, full) in enumerate(metabolic_abbrs):
        col = i % 2
        row = i // 2
        x_pos = metabolic_x - 0.07 + col * 0.14
        y_pos = start_y - 0.04 - row * y_spacing
        plt.text(x_pos, y_pos, f"{abbr}: {full}", 
                 ha='center', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
    
    # LPS Modification Pathway
    lps_x = 0.8
    lps_width = 0.2
    lps_height = len(lps_abbrs) * y_spacing + 0.04  # Extra space for header
    
    # Create border for LPS abbreviations
    lps_border = patches.FancyBboxPatch(
        (lps_x - lps_width/2, start_y - lps_height), 
        lps_width, lps_height,
        boxstyle=patches.BoxStyle("Round", pad=0.02),
        facecolor='white', alpha=0.8, edgecolor=colors['ushA'], linewidth=2
    )
    ax.add_patch(lps_border)
    
    # Add header
    plt.text(lps_x, start_y - 0.02, "LPS Modification", 
             ha='center', va='center', 
             fontsize=16, fontweight='bold', color=colors['ushA'])
    
    # Add abbreviations
    for i, (abbr, full) in enumerate(lps_abbrs):
        y_pos = start_y - 0.04 - i * y_spacing
        plt.text(lps_x, y_pos, f"{abbr}: {full}", 
                 ha='center', va='center', 
                 fontsize=14, fontweight='bold', color=colors['text'])
    
 
# Function to draw bacteria
def draw_bacteria(x, y, size=0.02, color=colors['bacteria']):
    bacteria = patches.Ellipse((x, y), size*2, size, 
                              fill=True, 
                              facecolor=color,
                              linewidth=1,
                              edgecolor='black')
    ax.add_patch(bacteria)

# Function to draw biofilm
def draw_biofilm(x, y, width=0.2, height=0.1, dense=True):
    # Draw biofilm matrix
    biofilm = patches.Ellipse((x, y), width, height, 
                             fill=True, 
                             facecolor=colors['biofilm'],
                             alpha=0.3,
                             linewidth=1,
                             edgecolor='black')
    ax.add_patch(biofilm)
    
    # Draw bacteria in biofilm
    bacteria_count = 12 if dense else 6
    for i in range(bacteria_count):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 0.8)
        bx = x + r * (width/2) * np.cos(angle)
        by = y + r * (height/2) * np.sin(angle)
        draw_bacteria(bx, by)

# Function to draw colistin molecules
def draw_colistin_molecules(x, y, width, height, count=5, penetration=True):
    for i in range(count):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.8, 1.2)
        cx = x + r * (width/2) * np.cos(angle)
        cy = y + r * (height/2) * np.sin(angle)
        
        # Draw colistin molecule
        colistin = patches.Circle((cx, cy), 0.015, 
                                 fill=True, 
                                 facecolor=colors['colistin'],
                                 linewidth=1,
                                 edgecolor='black',
                                 alpha=0.8)
        ax.add_patch(colistin)
        
        plt.text(cx, cy, "Colistin", 
                 ha='center', va='center', 
                 fontsize=11, fontweight='bold', color='white')
        
        # Draw penetration arrows for some colistin molecules
        if penetration and np.random.random() < 0.5:
            # Calculate direction toward biofilm center
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx**2 + dy**2)
            dx /= dist
            dy /= dist
            
            # Draw arrow showing penetration
            arrow = patches.FancyArrowPatch(
                (cx, cy), (cx + dx*0.03, cy + dy*0.03),
                connectionstyle="arc3,rad=0", 
                arrowstyle="Simple,head_width=4,head_length=4",
                color=colors['colistin'], linewidth=1, alpha=0.6
            )
            ax.add_patch(arrow)

# Function to draw colistin molecule
def draw_colistin(x, y, binding=True):
    colistin = patches.Circle((x, y), 0.02, 
                             fill=True, 
                             facecolor=colors['colistin'],
                             linewidth=1,
                             edgecolor='black',
                             alpha=0.8)
    ax.add_patch(colistin)
    
    plt.text(x, y, "Colisitn", 
             ha='center', va='center', 
             fontsize=11, fontweight='bold', color='white')
    
    if binding:
        # Draw binding interaction
        for i in range(5):
            angle = i * 2 * np.pi / 5
            dx = 0.01 * np.cos(angle)
            dy = 0.01 * np.sin(angle)
            
            line = patches.ConnectionPatch(
                (x+dx, y+dy), (x+dx*3, y+dy*3),
                "data", "data",
                arrowstyle="-",
                linestyle=":",
                color="purple",
                linewidth=1
            )
            ax.add_patch(line)

# Draw the LPS structure
def draw_lps(x, y, modified=False):
    # Draw lipid A
    lipid_a = patches.Circle((x, y), 0.015, 
                            fill=True, 
                            facecolor='#e67e22' if modified else '#d35400',
                            linewidth=1,
                            edgecolor='black')
    ax.add_patch(lipid_a)
    
    # Draw core oligosaccharide
    for i in range(3):
        sugar = patches.Circle((x, y+0.02+i*0.015), 0.01, 
                              fill=True, 
                              facecolor='#f1c40f' if modified else '#f39c12',
                              linewidth=1,
                              edgecolor='black')
        ax.add_patch(sugar)
    
    # Draw O-antigen
    if not modified:
        for i in range(5):
            o_sugar = patches.Circle((x, y+0.065+i*0.015), 0.008, 
                                    fill=True, 
                                    facecolor='#2ecc71',
                                    linewidth=1,
                                    edgecolor='black')
            ax.add_patch(o_sugar)
    else:
        for i in range(2):  # Shorter O-antigen in modified LPS
            o_sugar = patches.Circle((x, y+0.065+i*0.015), 0.008, 
                                    fill=True, 
                                    facecolor='#27ae60',
                                    linewidth=1,
                                    edgecolor='black')
            ax.add_patch(o_sugar)

# Define new cell position
cell_center_x = 0.4
cell_center_y = 0.6

# Draw the bacterial cell with membranes (no cell wall) - moved to upper left
# Outer membrane
outer_membrane = patches.Ellipse((cell_center_x, cell_center_y), 0.8, 0.6, 
                               fill=False, 
                               linewidth=3,
                               edgecolor=colors['outer_membrane'])
ax.add_patch(outer_membrane)

# Inner membrane
inner_membrane = patches.Ellipse((cell_center_x, cell_center_y), 0.7, 0.5, 
                               fill=True, 
                               facecolor=colors['cytoplasm'],
                               linewidth=2,
                               edgecolor=colors['inner_membrane'],
                               alpha=0.3)
ax.add_patch(inner_membrane)

# Add membrane labels (but not "Bacterial Cell" text)
plt.text(cell_center_x, cell_center_y + 0.28, "Outer Membrane", 
         ha='center', va='center', 
         fontsize=14, color=colors['outer_membrane'])

plt.text(cell_center_x, cell_center_y + 0.25, "Inner Membrane", 
         ha='center', va='center', 
         fontsize=14, color=colors['inner_membrane'])

# Create section headers - adjusted for new cell position
plt.text(cell_center_x - 0.25, cell_center_y + 0.15, "Biofilm Formation Pathway", 
         ha='center', va='center', 
         fontsize=14, fontweight='bold', color=colors['text'])

plt.text(cell_center_x, cell_center_y + 0.15, "Metabolic and Regulatory Pathways", 
         ha='center', va='center', 
         fontsize=14, fontweight='bold', color=colors['text'])

plt.text(cell_center_x + 0.25, cell_center_y + 0.15, "LPS Modification Pathway", 
         ha='center', va='center', 
         fontsize=14, fontweight='bold', color=colors['text'])

# Create the gene circles inside the cell - adjusted for new cell position
# Left quarter - Biofilm formation genes
create_gene_circle(cell_center_x - 0.3, cell_center_y + 0.1, "lsrR", colors['lsrR'])
create_gene_circle(cell_center_x - 0.25, cell_center_y + 0.1, "bdcA", colors['bdcA'])
create_gene_circle(cell_center_x - 0.2, cell_center_y + 0.1, "ratB", colors['ratB'])

# Middle half - Metabolic and Regulatory genes
# Top row - Metabolic genes
create_gene_circle(cell_center_x - 0.1, cell_center_y + 0.1, "leuD_2", colors['leuD_2'])
create_gene_circle(cell_center_x - 0.05, cell_center_y + 0.1, "idnK", colors['idnK'])
create_gene_circle(cell_center_x, cell_center_y + 0.1, "frdD", colors['frdD'])
create_gene_circle(cell_center_x + 0.05, cell_center_y + 0.1, "fepG", colors['fepG'])

# Bottom row - Regulatory genes
create_gene_circle(cell_center_x - 0.1, cell_center_y + 0.05, "punR", colors['punR'])
create_gene_circle(cell_center_x - 0.05, cell_center_y + 0.05, "rlmD", colors['rlmD'])
create_gene_circle(cell_center_x, cell_center_y + 0.05, "leuS", colors['leuS'])
create_gene_circle(cell_center_x + 0.05, cell_center_y + 0.05, "casA", colors['casA'])

# Right quarter - LPS modification genes
create_gene_circle(cell_center_x + 0.2, cell_center_y + 0.1, "ushA", colors['ushA'])
create_gene_circle(cell_center_x + 0.25, cell_center_y + 0.1, "rfaL", colors['rfaL'])
create_gene_circle(cell_center_x + 0.3, cell_center_y + 0.1, "rhmD", colors['rhmD'])

# Add mechanism abbreviations above gene circles - adjusted for new cell position
# Biofilm formation mechanisms
plt.text(cell_center_x - 0.3, cell_center_y + 0.13, "QS", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x - 0.25, cell_center_y + 0.13, "cGS", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x - 0.2, cell_center_y + 0.13, "SP", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Metabolic mechanisms (top row)
plt.text(cell_center_x - 0.1, cell_center_y + 0.13, "LB", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x - 0.05, cell_center_y + 0.13, "CM", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x, cell_center_y + 0.13, "AR", ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x + 0.05, cell_center_y + 0.13, "IA", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Regulatory mechanisms (bottom row) - placed below gene circles as requested
plt.text(cell_center_x - 0.1, cell_center_y + 0.02, "PM", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x - 0.05, cell_center_y + 0.02, "RM", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x, cell_center_y + 0.02, "PS", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x + 0.05, cell_center_y + 0.02, "GP", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# LPS modification mechanisms
plt.text(cell_center_x + 0.2, cell_center_y + 0.13, "USM", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x + 0.25, cell_center_y + 0.13, "OAA", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
plt.text(cell_center_x + 0.3, cell_center_y + 0.13, "RhM", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Create central points for converging pathways - adjusted for new cell position
left_central_x, left_central_y = cell_center_x - 0.25, cell_center_y - 0.05  # Biofilm formation
middle_central_x, middle_central_y = cell_center_x, cell_center_y - 0.05  # Metabolic/Regulatory
right_central_x, right_central_y = cell_center_x + 0.25, cell_center_y - 0.05  # LPS modification

# Create arrows for the left pathways (Biofilm formation) - adjusted for new cell position
# From lsrR gene - Quorum sensing
create_arrow_with_text(cell_center_x - 0.3, cell_center_y + 0.08, cell_center_x - 0.3, cell_center_y + 0.02, "", color=colors['lsrR'])

# From bdcA gene - c-di-GMP signaling
create_arrow_with_text(cell_center_x - 0.25, cell_center_y + 0.08, cell_center_x - 0.25, cell_center_y + 0.02, "", color=colors['bdcA'])

# From ratB gene - Surface properties
create_arrow_with_text(cell_center_x - 0.2, cell_center_y + 0.08, cell_center_x - 0.2, cell_center_y + 0.02, "", color=colors['ratB'])

# Converging arrows to central point
create_arrow_with_text(cell_center_x - 0.3, cell_center_y + 0.02, left_central_x, left_central_y, "", color=colors['lsrR'])
create_arrow_with_text(cell_center_x - 0.25, cell_center_y + 0.02, left_central_x, left_central_y, "", color=colors['bdcA'])
create_arrow_with_text(cell_center_x - 0.2, cell_center_y + 0.02, left_central_x, left_central_y, "", color=colors['ratB'])

# Central point to biofilm formation
create_arrow_with_text(left_central_x, left_central_y, cell_center_x - 0.25, cell_center_y - 0.15, "", color=colors['arrow'], connectionstyle="arc3,rad=-0.1")

# Add EBF label
plt.text(cell_center_x - 0.25, cell_center_y - 0.1, "EBF", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Biofilm to physical barrier
create_arrow_with_text(cell_center_x - 0.25, cell_center_y - 0.15, cell_center_x - 0.25, cell_center_y - 0.25, "", color=colors['arrow'], connectionstyle="arc3,rad=0")

# Add PBC label
plt.text(cell_center_x - 0.25, cell_center_y - 0.2, "PBC", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Create arrows for the middle pathways (Metabolic and Regulatory) - adjusted for new cell position
# From metabolic genes to processes
create_arrow_with_text(cell_center_x - 0.1, cell_center_y + 0.08, cell_center_x - 0.1, cell_center_y, "", color=colors['metabolic'])
create_arrow_with_text(cell_center_x - 0.05, cell_center_y + 0.08, cell_center_x - 0.05, cell_center_y, "", color=colors['metabolic'])
create_arrow_with_text(cell_center_x, cell_center_y + 0.08, cell_center_x, cell_center_y, "", color=colors['metabolic'])
create_arrow_with_text(cell_center_x + 0.05, cell_center_y + 0.08, cell_center_x + 0.05, cell_center_y, "", color=colors['transport'])

# From regulatory genes to processes
create_arrow_with_text(cell_center_x - 0.1, cell_center_y + 0.03, cell_center_x - 0.1, cell_center_y - 0.05, "", color=colors['regulatory'])
create_arrow_with_text(cell_center_x - 0.05, cell_center_y + 0.03, cell_center_x - 0.05, cell_center_y - 0.05, "", color=colors['translation'])
create_arrow_with_text(cell_center_x, cell_center_y + 0.03, cell_center_x, cell_center_y - 0.05, "", color=colors['translation'])
create_arrow_with_text(cell_center_x + 0.05, cell_center_y + 0.03, cell_center_x + 0.05, cell_center_y - 0.05, "", color=colors['genome'])

# Converging arrows to central point
create_arrow_with_text(cell_center_x - 0.1, cell_center_y - 0.05, middle_central_x, middle_central_y, "", color=colors['metabolic'])
create_arrow_with_text(cell_center_x - 0.05, cell_center_y - 0.05, middle_central_x, middle_central_y, "", color=colors['translation'])
create_arrow_with_text(cell_center_x, cell_center_y - 0.05, middle_central_x, middle_central_y, "", color=colors['translation'])
create_arrow_with_text(cell_center_x + 0.05, cell_center_y - 0.05, middle_central_x, middle_central_y, "", color=colors['genome'])

# Central point to membrane effects
create_arrow_with_text(middle_central_x, middle_central_y, cell_center_x, cell_center_y - 0.15, "", color=colors['arrow'], connectionstyle="arc3,rad=-0.1")

# Add AMC label
plt.text(cell_center_x, cell_center_y - 0.1, "AMC", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Membrane effects to permeability
create_arrow_with_text(cell_center_x, cell_center_y - 0.15, cell_center_x, cell_center_y - 0.25, "", color=colors['arrow'], connectionstyle="arc3,rad=0")

# Add MMP label
plt.text(cell_center_x, cell_center_y - 0.2, "MMP", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Create arrows for the right pathways (LPS modification) - adjusted for new cell position
# From ushA gene - UDP-sugar metabolism
create_arrow_with_text(cell_center_x + 0.2, cell_center_y + 0.08, cell_center_x + 0.2, cell_center_y + 0.02, "", color=colors['ushA'])

# From rfaL gene - O-antigen attachment
create_arrow_with_text(cell_center_x + 0.25, cell_center_y + 0.08, cell_center_x + 0.25, cell_center_y + 0.02, "", color=colors['rfaL'])

# From rhmD gene - Rhamnose metabolism
create_arrow_with_text(cell_center_x + 0.3, cell_center_y + 0.08, cell_center_x + 0.3, cell_center_y + 0.02, "", color=colors['rhmD'])

# Converging arrows to central point
create_arrow_with_text(cell_center_x + 0.2, cell_center_y + 0.02, right_central_x, right_central_y, "", color=colors['ushA'])
create_arrow_with_text(cell_center_x + 0.25, cell_center_y + 0.02, right_central_x, right_central_y, "", color=colors['rfaL'])
create_arrow_with_text(cell_center_x + 0.3, cell_center_y + 0.02, right_central_x, right_central_y, "", color=colors['rhmD'])

# Central point to modified LPS structure
create_arrow_with_text(right_central_x, right_central_y, cell_center_x + 0.25, cell_center_y - 0.15, "", color=colors['arrow'], connectionstyle="arc3,rad=-0.1")

# Add MLS label
plt.text(cell_center_x + 0.25, cell_center_y - 0.1, "MLS", ha='center', va='center', fontsize=14, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Modified LPS to reduced binding
create_arrow_with_text(cell_center_x + 0.25, cell_center_y - 0.15, cell_center_x + 0.25, cell_center_y - 0.25, "", color=colors['arrow'], connectionstyle="arc3,rad=0")

# Add RCB label
plt.text(cell_center_x + 0.25, cell_center_y - 0.2, "RCB", ha='center', va='center', fontsize=12, fontweight='bold', color=colors['text'],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

# Draw biofilm on the left side - adjusted for new cell position
draw_biofilm(cell_center_x - 0.25, cell_center_y - 0.2, width=0.15, height=0.08, dense=True)

# Draw LPS structures on the right side - adjusted for new cell position
draw_lps(cell_center_x + 0.2, cell_center_y - 0.2, modified=False)  # Normal LPS
draw_lps(cell_center_x + 0.3, cell_center_y - 0.2, modified=True)   # Modified LPS

# Add LPS labels
plt.text(cell_center_x + 0.2, cell_center_y - 0.25, "Normal LPS", 
         ha='center', va='center', 
         fontsize=13, color=colors['text'])

plt.text(cell_center_x + 0.3, cell_center_y - 0.25, "Modified LPS", 
         ha='center', va='center', 
         fontsize=13, color=colors['text'])

# Draw colistin molecules - adjusted for new cell position
# Around biofilm (left)
draw_colistin_molecules(cell_center_x - 0.25, cell_center_y - 0.2, 0.2, 0.1, count=6, penetration=False)

# Around membrane (middle)
draw_colistin_molecules(cell_center_x, cell_center_y - 0.2, 0.2, 0.1, count=6, penetration=False)

# Around LPS (right)
draw_colistin(cell_center_x + 0.2, cell_center_y - 0.1, binding=True)  # Binding to normal LPS
draw_colistin(cell_center_x + 0.3, cell_center_y - 0.1, binding=False) # Not binding to modified LPS

# Final arrows to colistin resistance
create_arrow_with_text(cell_center_x - 0.25, cell_center_y - 0.25, cell_center_x, cell_center_y - 0.35, "", color=colors['arrow'])
create_arrow_with_text(cell_center_x, cell_center_y - 0.25, cell_center_x, cell_center_y - 0.35, "", color=colors['arrow'])
create_arrow_with_text(cell_center_x + 0.25, cell_center_y - 0.25, cell_center_x, cell_center_y - 0.35, "", color=colors['arrow'])

# Add final resistance text
resistance_box = patches.FancyBboxPatch(
    (cell_center_x - 0.15, cell_center_y - 0.4), 0.3, 0.06, 
    boxstyle=patches.BoxStyle("Round", pad=0.02),
    facecolor='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5
)
ax.add_patch(resistance_box)

plt.text(cell_center_x, cell_center_y - 0.37, "Colistin Resistance", 
         ha='center', va='center', 
         fontsize=16, fontweight='bold', color='white')

# Display gene descriptions on the right side
display_genes_description()

# Display mechanism abbreviations below colistin resistance
display_mechanism_abbreviations_bottom()

# Remove axes
plt.axis('off')

# Save the figure
plt.tight_layout()
plt.savefig('e:/Evolution/2024/')
plt.close()



# In[ ]:





# In[ ]:




