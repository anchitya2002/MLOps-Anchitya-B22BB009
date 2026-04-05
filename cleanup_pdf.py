import re

with open('generate_pdf.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Everything after the conclusion paragraph should be cut + replaced with clean save block
pat = r'(        "results here are consistent with what has been reported in the literature\."\s+\)\s+\n).*'
replacement = r'\1\n    # ---- save --------------------------------------------------------\n    out = \'B22BB009_Anchitya_Ass5_v3.pdf\'\n    pdf.output(out)\n    print(f\'Report saved -> {out}\')\n\n\nif __name__ == \'__main__\':\n    build()\n'

new_content = re.sub(pat, replacement, content, flags=re.DOTALL)
if new_content == content:
    print('Pattern not matched!')
else:
    with open('generate_pdf.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f'Done. Total lines: {new_content.count(chr(10))}')
