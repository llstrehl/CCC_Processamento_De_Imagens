# Dados

Nada desta pasta é versionado: são muitas imagens e pesadas. Só este arquivo vai
para o repositório.

Organização esperada na máquina de quem for treinar:

```
data/
  fotos/        fotos originais dos lotes, tiradas pelo grupo
  recortes/
    inteiro/        recortes de grãos inteiros
    defeituoso/     recortes de grãos quebrados ou defeituosos
```

A pasta `recortes` é gerada automaticamente:

```bash
python -m src.dataset data/fotos --saida data/recortes
```

O programa já separa os recortes nas duas pastas usando o critério clássico. O
trabalho manual é conferir e mover os que ele classificou errado antes de treinar.
