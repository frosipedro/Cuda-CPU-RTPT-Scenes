# Ray Tracing & Path Tracing em CUDA/CPU (Single Thread - OpenMP)

Este projeto contém implementações de Ray Tracing e Path Tracing aceleradas por GPU usando CUDA. Inclui cenas procedurais, suporte a múltiplos materiais e luzes, além de exemplos de uso em CPU e GPU.

## Compilação

### Pré-requisitos

- CUDA Toolkit instalado
- Compilador compatível com CUDA (ex: nvcc)
- [stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h) na mesma pasta dos arquivos `.cu`

### Comando de compilação

#### Compilação automática (Makefile)

```sh
make
```

Os binários serão gerados na pasta `bin/`.

#### Limpar os arquivos gerados

```sh
make rm
```

## Execução

### Parâmetros importantes

- **samples**: Amostras por pixel. Aumenta a qualidade, mas o tempo cresce linearmente.
- **width/height**: Resolução da imagem.
  - Teste: 1280×720
  - Alta qualidade: 1920×1080 (1080p) ou 2560×1440 (1440p)
- **maxDepth**: Máximo de bounces (reflexões/refrações).
  - 4–8 é suficiente para cenas simples

### Exemplos de execução

```sh
./bin/raytracer
/.bin/rt_and_pt
./bin/surreal_pathtracer
```

Os programas pedem os parâmetros via terminal.

## Dicas de Performance

- Faça um render rápido com `samples=8` para checar composição.
- Para imagem final, aumente os samples.
- Compile com `-O3` e/ou `-use_fast_math` para mais velocidade (pode afetar precisão).
- O principal gargalo é o tempo de computação, não a memória.
  - Exemplo: 2560×1440 usa ~44MB para o framebuffer (floats).

## Resultados Esperados

- Com `samples=100` e `1280×720` em uma GPU moderna, a imagem já fica razoável (algum ruído).
- Para imagem limpa, aumente os samples.
- No CPU, o tempo de render é muito maior — a vantagem da GPU é enorme.

## Recursos Extras

Se quiser implementar mais recursos, veja sugestões:

- Luzes de área suave (soft area lights)
- Materiais dielétricos (vidro)
- BVH para aceleração (essencial para muitas esferas)
- Render progressivo (salva a cada passe)
- Denoiser

Abra uma issue ou peça ajuda para adicionar qualquer um desses itens!

---

## Créditos: 

- Cristian dos Santos Siquiera — https://github.com/CristianSSiqueira
- Pedro Rockenbach Frosi — https://github.com/frosipedro

**Uso do arquivo 'stb_image_write.h', disponibilizado no projeto:**
- https://github.com/nothings/stb

