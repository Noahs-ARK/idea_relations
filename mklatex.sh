#!/bin/bash

pdflatex $1
bibtex $1
pdflatex $1
pdflatex $1

