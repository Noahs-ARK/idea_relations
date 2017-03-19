# -*- coding: utf-8 -*-


def write_tex_file(filename, info):
    template_file = "templates/template.tex"
    with open(template_file) as fin:
        lines = fin.readlines()
        output = "".join(lines)
    with open(filename, "w") as fout:
        fout.write(output.format(**info))

