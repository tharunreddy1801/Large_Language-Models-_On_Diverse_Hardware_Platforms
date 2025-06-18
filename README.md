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
 ##  📁 Directory Structure
* ├── 📁 src/                                                  # Source code files
* ├── 📁 BroadRange_FPGA/                                      # work_experience
* ├── 📁 Models/                                               # Models
* ├── 📄 onnx and HuggingFace.ipynb                            # code
* ├── 📄 graphs.docx                                           # Visualisation
* ├── 📄 llm on Diverse hardware platforms_report.pdf          # report
* ├── 📄 README.md               # Project overview and setup
* └── 📄 LICENSE                 # License file
## 🏛️ Architecture
  * A[Hugging Face Model] --> B[ONNX Conversion]
  * B --> C[Hardware Optimization]
  * C --> D[CPU Deployment]
  * C --> E[GPU Deployment] 
  * C --> F[FPGA Deployment]
  * D,E,F --> G[Performance Analysis]
## Core Components

* **Model Layer**: Pre-trained transformers (BERT, TinyBERT)
* **Conversion Layer**: ONNX standardization for cross-platform compatibility
* **Hardware Layer**: Platform-specific optimizations
* **Analysis Layer**: Performance monitoring and visualization
## 🛠️ Tech Stack
***Core Technologies***

* **Python 3.8+**: Primary programming language
* **PyTorch**: Deep learning framework
* **ONNX**: Cross-platform model standardization
* **Hugging Face Transformers**: Pre-trained model library

***Hardware Platforms***

* **CPU**: Intel/AMD x86_64 architectures
* **GPU**: NVIDIA CUDA-enabled GPUs
* **FPGA**: Xilinx/Intel FPGA platforms (planned)

***Development Tools***

* **Jupyter Notebooks**: Interactive development
* **Matplotlib/Seaborn**: Data visualization
* **Netron**: Neural network visualization
* **Git**: Version control
## ⚙️ Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/tharunreddy1801/Large_Language-Models-_On_Diverse_Hardware_Platforms.git
cd Large_Language-Models-_On_Diverse_Hardware_Platforms
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, use:
```bash
pip install torch transformers onnx onnxruntime huggingface-hub numpy pandas matplotlib seaborn jupyter notebook
```
### 3. Download & Organize Models
Download models from Hugging Face and place them like:
```bash
models/
├── bert-base-uncased/
├── bert-large-uncased/
├── distilbert-base-uncased/
└── huawei-noah--TinyBERT_General_4L_312D/
```
### 4. Setup Hardware Platforms
For CPU Platform:
```bash
pip install onnx
pip install onnxruntime
```
For GPU(Paper Space[Vm]) Platform:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```
For FPGA Platform:
```bash
pip install pynq
```
### 5. Verify Installation
```bash
python scripts/verify_setup.py
```
### 6. Convert Models to ONNX
```bash
python convert_models.py --model bert-base-uncased --output models/onnx/
```
### 7. Run Performance Benchmarks
```bash
python benchmark.py --platforms cpu,gpu,fpga --models all
```
### 8. Start Jupyter Notebook (Optional)
```bash
jupyter notebook
```
### 9. Quick Test Run
```bash
python quick_test.py --model bert-base-uncased --platform cpu
```
## 📈 Performance Results
* Benchmark Summary (Input: "What is capital of India?")
![image](https://github.com/user-attachments/assets/2f1baff8-8745-446d-be39-879d13f9b70e)

* 🎉 **Key Achievement**: ONNX optimization provides up to 9.1x speedup on GPU and 5.8x speedup on CPU
## Energy Efficiency Metrics
![image](https://github.com/user-attachments/assets/249f04e8-4c4c-433b-81ad-b2199cb24fb2)
* Estimated based on research analysis
## 🔮 Future Scope

 * **FPGA Implementation**: Complete FPGA deployment and optimization
 * **Quantum Computing**: Explore quantum-classical hybrid approaches
 * **Edge Computing**: Mobile and IoT device optimization
 * **AutoML Integration**: Automated hardware-software co-design
 * **Real-time Applications**: Chatbot and voice assistant implementations
 * **Security Features**: Privacy-preserving LLM deployment
## 🙏 Acknowledgments

* Hugging Face for pre-trained models
* ONNX community for optimization tools
* Xilinx  for FPGA development tools
* PyTorch and TensorFlow communities

## 📚 References

* ONNX Documentation
* Hugging Face Transformers
* FlightLLMs
* Hardware-Aware Model Optimization
## 👨‍💻 Author
Y.Sai Tharun Reddy

