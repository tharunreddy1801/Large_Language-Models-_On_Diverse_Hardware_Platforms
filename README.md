# Large_Language-Models-_On__Diverse_Hardware_Platforms
I have deployed Existing Hugging Face model on Diverse Hardware platforms to reduce Latency, and increase throughput. FPGA works on Parallelism in which it's computational Power will be reduce by deploying FPGA
## 🔍 Problem Statement
With the exponential growth of LLMs, efficient deployment across diverse hardware platforms has become crucial for:
* **Scalability** across different computational environments
* **Cost optimization** for enterprise deployments
* **Energy efficiency** for sustainable AI practices
* **Real-time applications** requiring low-latency inference
## ✨ Key Features
* 🏗️ **Multi-Hardware Support**: CPU, GPU, and FPGA implementations
* 🔄 **Model Conversion Pipeline**: Hugging Face → ONNX → Hardware-specific optimization
* ⚡ **Performance Benchmarking**: Comprehensive latency and throughput analysis
* 📊 **Visualization Tools**: Real-time performance monitoring and comparison
* 🎯 **Model Optimization**: Quantization and compression techniques
* 🌐 **Framework Agnostic**: Support for multiple ML frameworks give this lines in readme code
## 📈 Performance Results
* Benchmark Summary (Input: "What is capital of India?")
![image](https://github.com/user-attachments/assets/2f1baff8-8745-446d-be39-879d13f9b70e)

* 🎉 **Key Achievement**: ONNX optimization provides up to 9.1x speedup on GPU and 5.8x speedup on CPU
## 🏛️ Architecture
  graph TD
  * A[Hugging Face Model] --> B[ONNX Conversion]
  * B --> C[Hardware Optimization]
  * C --> D[CPU Deployment]
  * C --> E[GPU Deployment] 
  * C --> F[FPGA Deployment]
  * D,E,F --> G[Performance Analysis]
