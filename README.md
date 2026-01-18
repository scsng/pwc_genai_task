# GenAI based Legal help PwC project

## Running the application

> [!CAUTION]
>
> The system was developed and tested on a Linux system (Ubuntu). Cannot guarantee solid running on other environments.

1. Install the required components:

   1. First install the Docker engine & Docker Compose. You can find a description here for that.
   2. Install uv package manager. You can find a description for that here.

2. There are two .env.example files, in the main & in the deployment folder. 

   1. Copy them as .env files.
   2. You can use your HuggingFace token, or you can use my temporary one, I sent in the email.

3. Run the [./deployment/docker-compose.yaml](./deployment/docker-compose.yaml ) file.
4. ```bash
   cd deployment
   sh vllm_run.sh
   ```

   Choose to run VLLM via its pip package and CLI instead of the Docker package, because I had very  imited amout of VRAM, and the container seemed it will neeed more than the avaliable (I didn't have time doing a deeper dive in it.pwc_task)

4. ```bash
   uv sync
   uv run streamlit run app.py
   ```

5. Open the link of the streamlit app.


Tool caht template is from officakl vllm with a little twist, I choose this becouse big, but fit into memory, and has tool usage







## Testing

I run a quality and performance test. The test notebook is available here: [tests.ipynb](testing/tests.ipynb)

### Quality test

[Input file](testing/input/quality.csv), [Output file](testing/output/quality.csv)

1. Because of the limited amount of time, and ~objective results I used LLM as a judge for the test evaluation . I used Openai GPT-5.2 model for this.

### Performance test

[Input file](testing/input/performance.csv), [Output file](testing/output/performance.csv)

