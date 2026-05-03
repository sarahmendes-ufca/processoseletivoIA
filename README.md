# 📝 Relatório final

> 👤 **Informações**
> - **Nome**: Sarah Mendes Teles 
> - **Curso**: Engenharia de Software 
> - **Semestre**: 2026.1

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

---

## 1️⃣ Resumo da Arquitetura do Modelo
Estrutura de diretórios do projeto: 

```plaintext
processoseletivoIA/
├── .github/                # Github CI/CD (não modificar)
│   └── workflows/
│       └── ci.yml            
├── .devcontainer/          # Devconteiner (não modificar)
│   └── devcontainer.json
├── model.h5                # Modelo não-otimizado
├── model.tflite            # Modelo otimizado
├── train_model.py          # Código python para gerar o modelo 
├── optimize_model.py       # Código para otimizar o modelo
├── requirements.txt        # Bibliotecas
├── .gitignore              # Arquivos/diretórios ignorados
└── README.md               # Documentção
```
A escolha de uma arquitetura simples visa:

- Compatibilidade com dispositivos de baixo poder computacional
- Execução em pipelines automatizados (CI/CD)
- Facilidade de conversão e otimização para TFLite

Modelos mais complexos poderiam aumentar a acurácia, porém comprometeriam a eficiência, tornando-os menos adequados para Edge AI.

### 1. 🔄 Fluxo dos Dados
As imagens do MNIST possuem dimensão 28x28 em escala de cinza. Antes de serem utilizadas pelo modelo, passam por etapas de pré-processamento:
```bash 
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```
- **Normalização:** os valores dos pixels são convertidos de [0,255] para [0,1], facilitando o treinamento.
- **Expansão de dimensão:** adiciona o canal (1), necessário para camadas convolucionais.

Além disso, foi utilizada apenas uma amostra reduzida do dataset para tornar o treinamento mais rápido e leve:
```bash 
x_train = x_train[:10000]
y_train = y_train[:10000]
```
Essa escolha está alinhada com o objetivo de simular um cenário com restrições de recursos.

### 2. 🧠 Arquitetura da CNN
O modelo foi construído utilizando a API Sequential do Keras:
```bash 
model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])
```
Descrição das camadas:

- **Conv2D (16 filtros):** extrai características básicas da imagem (bordas e formas simples)
- **MaxPooling:** reduz dimensionalidade e custo computacional
- **Conv2D (32 filtros):** extrai características mais complexas
- **Flatten:** transforma os dados em vetor
- **Dense (Softmax):** realiza a classificação final (10 classes)

A arquitetura foi mantida propositalmente simples para reduzir custo computacional e facilitar a conversão para dispositivos embarcados.

### 3. ⚙️ Treinamento
O modelo foi compilado com:
```bash 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- **Adam:** escolhido pela rápida convergência
- **Sparse categorical crossentropy:** adequado para classificação multiclasse com rótulos inteiros

Treinamento:
```bash
model.fit(
    x_train,
    y_train,
    epochs=3,
    validation_data=(x_test, y_test)
)
```
O número reduzido de épocas foi uma decisão intencional para manter baixo tempo de execução e consumo de recursos.

### 4. 💾 Salvamento do modelo
Após o treinamento, o modelo foi salvo no formato Keras:
```bash
model.save("model.h5")
```
Esse formato permite fácil reutilização e posterior conversão.


## 2️⃣ Bibliotecas Utilizadas

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy"/>
</div>

> *TensorFlow: >=2.12*


## 3️⃣ Técnica de Otimização do Modelo

O modelo foi convertido para TensorFlow Lite com o objetivo de execução em dispositivos embarcados:
```bash
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```
A técnica utilizada foi:


- **Dynamic Range Quantization:** Essa técnica reduz o tamanho do modelo convertendo os pesos para representações mais 
eficientes, sem necessidade de dataset adicional.

### 3. Benefícios da Otimização
- Redução do tamanho do modelo  
- Menor consumo de memória  
- Inferência mais rápida  
- Compatibilidade com dispositivos embarcados  


## 4️⃣ Resultados Obtidos

### 1. 📊 Métricas do modelo
O desempenho do modelo foi avaliado utilizando múltiplas métricas, permitindo uma análise mais completa da sua eficiência:

- **Acurácia (Accuracy):** ~97% no conjunto de teste
- **Loss (Sparse Categorical Crossentropy):** valor baixo e estável ao final do treinamento
- **Acurácia de validação:** próxima da acurácia de treino

#### 🔍 Análise das métricas

A alta acurácia indica que o modelo é capaz de classificar corretamente a maioria dos dígitos manuscritos.

O valor reduzido da função de perda (loss) demonstra que o modelo não apenas acerta as previsões, mas também está confiante nas classificações realizadas.

A proximidade entre a acurácia de treino e validação sugere:

- Boa capacidade de generalização
- Ausência de overfitting significativo

Mesmo com um número reduzido de épocas e uma arquitetura simplificada, o modelo apresentou desempenho consistente, reforçando sua eficiência.

---
### 2. ⚖️ Comparação: Tamanho x desempenho
| **Modelo** | **Tamanho**| **Acurácia** | | **Eficiência** |
| :--- | :---: | :---: | ---: |
| Modelo original (.h5) | ~186 KB | ~97% | | Alto custo relativo |
| Modelo otimizado (.tflite) | ~18 KB | ~96–97% | | Alta eficiência |

#### 📉 Análise do trade-off

A aplicação de quantização resultou em:

- Redução de aproximadamente 90% no tamanho do modelo
- Manutenção de uma acurácia praticamente equivalente

Isso evidencia um trade-off altamente favorável:

- Pequena (ou desprezível) perda de precisão
- Grande ganho em eficiência, portabilidade e uso de memória

---
### 3. 🚀 Ganho com Otimização
A conversão para TensorFlow Lite com quantização permitiu:

- Execução mais rápida em dispositivos com baixo poder computacional
- Redução do consumo de memória em tempo de inferência
- Maior viabilidade para aplicações embarcadas e IoT

Mesmo após a otimização, o modelo manteve desempenho consistente, demonstrando que técnicas de compressão podem ser aplicadas sem comprometer significativamente a qualidade das previsões.

## 5️⃣ Comentários Adicionais

### 🔥 Dificuldades encontradas
As maiores dificuldades encontradas durante o desenvolvimento do projeto foram os cuidados com a otimização:

- Durante o projeto, tive a experiência de desenvolver uma aplicação de IA com a biblioteca TensorFlow. Contúdo, o projeto necessita de um desenvolvimento mais cuidadoso com os modelos por se tratar de Edge AI. A etapa de conversão do modelo para TFLite gerou alguns avisos e mensagens que precisaram ser analisados para garantir que não eram erros críticos e que o processo estava sendo concluído corretamente.

### 🌟 Decisões técnicas importantes

- **Uso de uma CNN simples**: O modelo foi projetado com poucas camadas convolucionais para garantir baixo custo computacional e compatibilidade com ambientes restritos, como pipelines de CI e dispositivos Edge.

- **Priorização da simplicidade**: O projeto foi desenvolvido com foco em confiabilidade e execução automática, evitando dependências desnecessárias ou configurações complexas.

### ❗ Limitações do modelo

- **Modelo simplificado**: A arquitetura foi mantida simples para atender às restrições de execução, o que pode limitar o desempenho em cenários mais complexos.

- **Treinamento com poucas épocas**: O número reduzido de épocas pode impedir que o modelo atinja seu máximo potencial de acurácia.

- **Possível perda mínima de precisão após otimização**: A conversão para TFLite com quantização pode causar pequenas perdas de precisão, embora não significativas neste caso.

### 🧠 Aprendizados durante o desenvolvimento

As dificuldades enfrentadas contribuíram para uma melhor compreensão de:
- Execução de modelos em ambientes isolados (Docker)
- Otimização de modelos para Edge AI
- Construção de pipelines automatizados
