"""Convert README.qmd to GitHub-compatible README.md.

Converts $$ display math to ```math code fences (bulletproof on GitHub),
fixes \\[4pt], \texttt, collapses double blank lines.
"""
import re

with open('README.qmd', 'r') as f:
    content = f.read()

# 1. Convert $$...$$ display math blocks to ```math fences
def convert_display_math(m):
    inner = m.group(1).strip()
    # Fix \\[4pt] -> \\ inside math
    inner = re.sub(r'\\\\(?:\[\d+pt\])', r'\\\\', inner)
    # Fix \texttt -> \mathtt
    inner = inner.replace(r'\texttt', r'\mathtt')
    return '\n```math\n' + inner + '\n```\n'

# Match $$ blocks (on their own lines, possibly with blank lines around)
content = re.sub(r'\n\$\$\n(.*?)\n\$\$', convert_display_math, content, flags=re.DOTALL)

# 2. Collapse runs of 3+ newlines down to 2 (one blank line)
content = re.sub(r'\n{3,}', '\n\n', content)

# 3. Remove any leftover audit script reference
# (not needed)

with open('README.md', 'w') as f:
    f.write(content)

print("Done! README.md generated.")
