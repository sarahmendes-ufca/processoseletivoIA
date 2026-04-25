# 📝 Relatório final

> 👤 **Informações**
> - **Nome**: Sarah Mendes Teles 
> - **Curso**: Engenharia de Software 
> - **Semestre**: 2026.1

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

---

### 1️⃣ Resumo da Arquitetura do Modelo
Estrutura de diretórios do projeto: 

```plaintext
processoseletivoIA/
├── .github/
│   └── workflows/
│       └── ci.yml            
├── .devcontainer/            
│   └── devcontainer.json
├── model.h5
├── model.tflite
├── train_model.py
├── optimize_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

Este projeto utiliza uma Rede Neural Convolucional (CNN) para a classificação de dígitos manuscritos do dataset MNIST.

#### 1. Fluxo dos Dados
As imagens de entrada possuem dimensão 28x28 em escala de cinza. Antes de serem processadas pelo modelo, elas são normalizadas e reorganizadas para o formato adequado da CNN.

#### 2. Arquitetura da CNN
O modelo é composto pelas seguintes camadas:

- Camada Conv2D com 16 filtros (3x3) e ativação ReLU  
- Camada MaxPooling2D (2x2)  
- Camada Conv2D com 32 filtros (3x3) e ativação ReLU  
- Camada MaxPooling2D (2x2)  
- Camada Flatten (achatamento dos dados)  
- Camada Dense com 10 neurônios e ativação Softmax (classificação das classes de 0 a 9)

Essa arquitetura foi projetada para ser leve e eficiente, adequada para aplicações de Edge AI.

#### 3. Treinamento
O modelo foi treinado utilizando o otimizador Adam e a função de perda sparse categorical crossentropy, durante poucas épocas, priorizando eficiência e compatibilidade com ambientes restritos.

#### 4. Pipeline de Deploy (Edge AI)
Após o treinamento, o modelo passa pelas seguintes etapas:

- Salvamento no formato Keras (.h5)  
- Conversão para TensorFlow Lite (.tflite)  
- Aplicação de quantização (Dynamic Range Quantization)

Esse processo reduz o tamanho do modelo e o torna adequado para execução em dispositivos embarcados e sistemas IoT.

### 2️⃣ Bibliotecas Utilizadas

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy"/>
</div>

- **TensorFlow:** >=2.12


### 3️⃣ Técnica de Otimização do Modelo

#### 1. Conversão para TensorFlow Lite
O modelo treinado em formato Keras (.h5) foi convertido para TensorFlow Lite (.tflite), um formato otimizado para inferência em dispositivos edge.

#### 2. Técnica de Otimização
Foi aplicada a técnica de **Dynamic Range Quantization**, que reduz o tamanho do modelo ao converter os pesos para uma representação mais eficiente, sem necessidade de dataset adicional.
---
#### 3. Benefícios da Otimização
- Redução do tamanho do modelo  
- Menor consumo de memória  
- Inferência mais rápida  
- Compatibilidade com dispositivos embarcados  

####  4. Resultado Final
O modelo otimizado mantém boa precisão enquanto se torna mais leve e eficiente para ambientes com restrição de recursos


### 4️⃣ Resultados Obtidos

#### 1. Desempenho do Modelo
O modelo apresentou bom desempenho na tarefa de classificação de dígitos manuscritos:

- Acurácia final: ~97%**

#### 2. Tamanho dos Modelos
Após o treinamento e a otimização, foram obtidos os seguintes arquivos:

- Modelo original (.h5): ~186 KB  
- Modelo otimizado (.tflite): ~18 KB  

#### 3. Ganho com Otimização
A conversão para TensorFlow Lite com quantização resultou em:

- Redução de ~90% do tamanho do modelo
- Menor consumo de memória  
- Maior eficiência para execução em dispositivos com recursos limitados  

### 5️⃣ Comentários Adicionais

#### 🔥 Dificuldades encontradas
As maiores dificuldades encontradas durante o desenvolvimento do projeto foram os cuidados com a otimização:

- Durante o projeto, tive a experiência de desenvolver uma aplicação de IA com a biblioteca TensorFlow. Contúdo, o projeto necessita de um desenvolvimento mais cuidadoso com os modelos por se tratar de Edge AI. A etapa de conversão do modelo para TFLite gerou alguns avisos e mensagens que precisaram ser analisados para garantir que não eram erros críticos e que o processo estava sendo concluído corretamente.

#### 🌟 Decisões técnicas importantes

- **Uso de uma CNN simples**: O modelo foi projetado com poucas camadas convolucionais para garantir baixo custo computacional e compatibilidade com ambientes restritos, como pipelines de CI e dispositivos Edge.

- **Priorização da simplicidade**: O projeto foi desenvolvido com foco em confiabilidade e execução automática, evitando dependências desnecessárias ou configurações complexas.

#### ❗ Limitações do modelo

- **Modelo simplificado**: A arquitetura foi mantida simples para atender às restrições de execução, o que pode limitar o desempenho em cenários mais complexos.

- **Treinamento com poucas épocas**: O número reduzido de épocas pode impedir que o modelo atinja seu máximo potencial de acurácia.

- **Possível perda mínima de precisão após otimização**: A conversão para TFLite com quantização pode causar pequenas perdas de precisão, embora não significativas neste caso.

#### 🧠 Aprendizados durante o desenvolvimento

As dificuldades enfrentadas contribuíram para uma melhor compreensão de:
- Execução de modelos em ambientes isolados (Docker)
- Otimização de modelos para Edge AI
- Construção de pipelines automatizados
