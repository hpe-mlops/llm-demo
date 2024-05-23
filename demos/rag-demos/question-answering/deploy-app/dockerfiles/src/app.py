import json
import requests

import os
import subprocess

import gradio as gr
from theme import EzmeralTheme
#from ezm import *
import ezm


DOMAIN_NAME = "svc.cluster.local"
NAMESPACE = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r").read()
DEPLOYMENT_NAME = "llm"
MODEL_NAME = DEPLOYMENT_NAME
SVC = f"{DEPLOYMENT_NAME}-transformer.{NAMESPACE}.{DOMAIN_NAME}"
URL = f"https://{SVC}/v1/models/{MODEL_NAME}:predict"

HEADER = """
    <div style='text-align: center;'>
        <img src='/file=image/app-header.png' alt='ai-enabled-search' 
             style='max-width: 100%; margin-left: auto; margin-right: auto;
             height: auto;'>
    </div>
"""

STYLE = """
    <style>
        .border-left-success {border-left: .25rem solid #1cc88a !important;padding-left: 5px;} 
        .border-left-ready {border-left: .25rem solid #A6A6A6 !important;padding-left: 5px;}
        .substep {margin-left: 50px;}
        .mainstep {display: flex;}

        .state.success { background: #01A982;}
        .state.ready { background: #ddd;}
        .state { margin-right: 5px;height: 14px;min-width: 14px;line-height: 14px;display: inline-block;color: #000;border-radius: 20px;text-indent: -9999px;font-size: .7em;text-shadow: 1px 1px 0 #fff;text-align: center;padding-right: 2px;box-shadow: inset 0 0 0 #000;}

         .blink { animation: blinker 1.5s linear infinite; } @keyframes blinker { 50% {opacity: 0;}}
    </style>
"""

CUSTOM_JS = """

<script>
function progress(lastIdex) {

	document.querySelector("#btn_interface").click()
	var strProgress = document.querySelector("#tb_interface > * > textarea").value
	var listProgress = strProgress.split("\\n")
	
	var pointerIndex = lastIdex;

	for (i = lastIdex; i < listProgress.length; i++) {
		var line = listProgress[i].trim();
		if(line === null || line === "")
			continue;
		pointerIndex++;
		
		var lineSplit = line.split(";")
		
		var elemDiv = document.querySelector("#"+lineSplit[0]);
		if(elemDiv === null)
			continue;
			
        if(elemDiv.classList.contains("mainstep")) {
            elemState = elemDiv.querySelector("span");
            if(lineSplit[1] === "OK") {
                elemState.classList.remove("ready");
                elemState.classList.add("success");
                elemDiv.classList.remove("blink");
            } else if(lineSplit[1] === "RUNNING") {
                elemDiv.classList.add("blink");
            }

        } else if( elemDiv.classList.contains("substep")) {
            elemDiv.classList.remove("border-left-ready")
            if(lineSplit[1] === "OK") {
                elemDiv.classList.add("border-left-success")
            }
        }
	} 
	return pointerIndex;
}

// remove async 
async function pollProgress () { 
  const timer = ms => new Promise(res => setTimeout(res, ms))
  await timer(3000);
  var lastIdex = 0;
  for (var i = 0; i < 3600; i++) {
	lastIdex = progress(lastIdex);
    await timer(1000); 
  }
  return "";
}

//window.onload = function(){
pollProgress();
//}
</script>
"""

## config ##

ezm.domain = 'ua.ezm.host'
ezm.username = NAMESPACE
#ezm.username = 'leejj881229'
ezm.password = 'hpinvent!23'

ezm.predictor_image = 'marketplace.us1.greenlake-hpe.com/ezmeral/ezkf/qna-vectorstore:v1.3.0-e658264'
ezm.llm_image = 'marketplace.us1.greenlake-hpe.com/ezmeral/ezkf/qna-llm:v1.3.0-e658264'
ezm.transformer_image = 'marketplace.us1.greenlake-hpe.com/ezmeral/ezkf/qna-transformer:v1.3.0-f7315d3'


ezm.llm_inference_service_name = "llm"
ezm.llm_predictor_name = "kserve-container"
ezm.llm_predictor_image = "marketplace.us1.greenlake-hpe.com/ezmeral/ezkf/qna-llm:v1.3.0-e658264"
ezm.llm_predictor_cpu_request = "1000m"
ezm.llm_predictor_cpu_limit = "1000m"
ezm.llm_predictor_memory_request = "8Gi"
ezm.llm_predictor_memory_limit = "8Gi"
ezm.llm_predictor_gpu_request= ""
ezm.llm_predictor_gpu_limit = ""

ezm.llm_transformer_name = "kserve-container"
ezm.llm_transformer_image = "marketplace.us1.greenlake-hpe.com/ezmeral/ezkf/qna-transformer:v1.3.0-f7315d3"
ezm.llm_transformer_cpu_request = "500m"
ezm.llm_transformer_cpu_limit = "500m"
ezm.llm_transformer_memory_request = "1Gi"
ezm.llm_transformer_memory_limit = "1Gi"
ezm.llm_transformer_gpu_request = ""
ezm.llm_transformer_gpu_limit = ""




progress = ""
IS_RUNNING = False

def deploy():
    global IS_RUNNING
    if(IS_RUNNING):
        print("Deployment is already underway.")
        return
    else:
        IS_RUNNING = True

    print("Deploy")
    set_config()
    ezm.config_environ()

    #ezm.get_token()
    create_vectorstore()
    serve_vectorstore()
    serve_llm()

    IS_RUNNING = False

def set_config():
    ezm.domain = input_domain.value
    ezm.username = input_username.value
    ezm.password = input_password.value
        
    ezm.predictor_image = input_predictor_image.value
    #ezm.llm_image = input_llm_image.value
    #ezm.transformer_image = input_transformer_image.value

    # llm config
    ezm.llm_inference_service_name = input_llm_inference_service_name.value
    ezm.llm_predictor_name = input_llm_predictor_name.value
    ezm.llm_predictor_image = input_llm_predictor_image.value
    ezm.llm_predictor_cpu_request = input_llm_predictor_cpu_request.value
    ezm.llm_predictor_cpu_limit = input_llm_predictor_cpu_limit.value
    ezm.llm_predictor_memory_request = input_llm_predictor_memory_request.value
    ezm.llm_predictor_memory_limit = input_llm_predictor_memory_limit.value
    ezm.llm_predictor_gpu_request = input_llm_predictor_gpu_request.value
    ezm.llm_predictor_gpu_limit = input_llm_predictor_gpu_limit.value

    ezm.llm_transformer_name = input_llm_transformer_name.value
    ezm.llm_transformer_image = input_llm_transformer_image.value
    ezm.llm_transformer_cpu_request = input_llm_transformer_cpu_request.value
    ezm.llm_transformer_cpu_limit = input_llm_transformer_cpu_limit.value
    ezm.llm_transformer_memory_request = input_llm_transformer_memory_request.value
    ezm.llm_transformer_memory_limit = input_llm_transformer_memory_limit.value
    ezm.llm_transformer_gpu_request = input_llm_transformer_gpu_request.value
    ezm.llm_transformer_gpu_limit = input_llm_transformer_gpu_limit.value



def log_progress(pro):
    global progress
    progress+= pro + "\n"


def create_vectorstore():
    log_progress("cvs;RUNNING")

    docs = ezm.load_docs("documents")
    log_progress("cvs_loading;OK")
    
    texts = ezm.process_docs(docs, chunk_size=500, chunk_overlap=0)
    log_progress("cvs_chunking;OK")

    embeddings = ezm.hugging_face_embeddings()
    log_progress("cvs_embedding;OK")

    db = ezm.storing_chroma(texts, embeddings)
    log_progress("cvs_storing;OK")

    log_progress("cvs;OK")

def serve_vectorstore():
    log_progress("svs;RUNNING")

    uri = ezm.logging_vector_store_artifact()
    log_progress("svs_logging_vs;OK")
    
    ezm.apply_vectorstore(uri)
    log_progress("svs_creating_service;OK")

    log_progress("svs;OK")

def serve_llm():
    log_progress("sll;RUNNING")

    ezm.apply_llm()
    log_progress("sll_creating_service;OK")

    log_progress("sll;OK")


def read_progress_call_by_client():
    return progress

def clean():
    global progress
    progress = ""
    ezm.delete_object("inferenceservices", "llm")
    ezm.delete_object("inferenceservices", "vectorstore")

if __name__ == "__main__":
    with gr.Blocks(theme=EzmeralTheme(), head=CUSTOM_JS) as app:
        gr.HTML(STYLE)
        # Application Header
        gr.HTML(HEADER)


        with gr.Accordion("General Info", open=False):
            input_domain = gr.Textbox(label="domain", value=ezm.domain)
            input_username = gr.Textbox(label="username", value=ezm.username)
            input_password = gr.Textbox(label="password", value=ezm.password)       
            input_predictor_image = gr.Textbox(label="predictor image", value=ezm.predictor_image)
            input_llm_image = gr.Textbox(label="llm image", value=ezm.llm_image)
            input_transformer_image = gr.Textbox(label="transformer image", value=ezm.transformer_image)

        with gr.Accordion("Inference Service Details", open=False):
            #gr.HTML("<div style=\"display: flex;justify-content: center;align-items: center;font-weight: var(--weight-bold);font-size: var(--text-xxl);\">Inference Service Details</div>")
            #with gr.Row():
            #    gr.Label("", show_label=False, container=False)
            #    input_inference_service_name = gr.Textbox(label="Name", value="", scale=4)
            #    gr.Label("", show_label=False, container=False)

            with gr.Row():
                #gr.HTML("<div style=\"display: flex;justify-content: center;align-items: center;font-weight: var(--weight-bold);font-size: var(--text-xxl);\">Inference Service Details</div>")
                gr.Label("Inference Service Details", show_label=False, container=False)
                gr.HTML()
                gr.HTML()
            with gr.Row():
                input_llm_inference_service_name = gr.Textbox(label="Name", value=ezm.llm_inference_service_name, scale=2)
                gr.Label("", show_label=False, container=False)

            with gr.Row():
                with gr.Column():
                    gr.Label("Predictor", show_label=False, container=False)
                    input_llm_predictor_name = gr.Textbox(label="Name", value=ezm.llm_predictor_name)
                    input_llm_predictor_image = gr.Textbox(label="Image", value=ezm.llm_predictor_image)
                    with gr.Row():
                        input_llm_predictor_cpu_request = gr.Textbox(label="CPU requests", value=ezm.llm_predictor_cpu_request)
                        input_llm_predictor_cpu_limit = gr.Textbox(label="CPU limits", value=ezm.llm_predictor_cpu_limit)
                    with gr.Row():
                        input_llm_predictor_memory_request = gr.Textbox(label="Memory requests", value=ezm.llm_predictor_memory_request)
                        input_llm_predictor_memory_limit = gr.Textbox(label="Memory limits", value=ezm.llm_predictor_memory_limit)
                    with gr.Row():
                        input_llm_predictor_gpu_request = gr.Textbox(label="GPU requests", value="")
                        input_llm_predictor_gpu_limit = gr.Textbox(label="GPU limits", value="")
                with gr.Column():
                    gr.Label("Transformer", show_label=False, container=False)
                    input_llm_transformer_name = gr.Textbox(label="Name", value=ezm.llm_transformer_name)
                    input_llm_transformer_image = gr.Textbox(label="Image", value=ezm.llm_transformer_image)

                    with gr.Row():
                        input_llm_transformer_cpu_request = gr.Textbox(label="CPU requests", value=ezm.llm_transformer_cpu_request)
                        input_llm_transformer_cpu_limit = gr.Textbox(label="CPU limits", value=ezm.llm_transformer_cpu_limit)
                    with gr.Row():
                        input_llm_transformer_memory_request = gr.Textbox(label="Memory requests", value=ezm.llm_transformer_memory_request)
                        input_llm_transformer_memory_limit = gr.Textbox(label="Memory limits", value=ezm.llm_transformer_memory_limit)
                    with gr.Row():
                        input_llm_transformer_gpu_request = gr.Textbox(label="GPU requests", value=ezm.llm_transformer_gpu_request)
                        input_llm_transformer_gpu_limit = gr.Textbox(label="GPU limits", value=ezm.llm_transformer_gpu_limit)
                with gr.Column():
                    gr.Label("Framework for model serving", show_label=False, container=False)
                    



        # interface
        tb_interface = gr.Textbox(value="", elem_id="tb_interface", visible=False)
        btn_interface = gr.Button(value="", elem_id="btn_interface", visible=False)
        btn_interface.click(fn=read_progress_call_by_client, outputs=tb_interface)

        btn_deploy = gr.Button("Deploy")
        #btn_deploy.click(fn=deploy, js="pollProgress")
        btn_deploy.click(fn=deploy)

        lb_create_vs = gr.HTML("<div id=\"cvs\" class=\"mainstep\"><span class=\"state ready\"></span>01.create-vectorstore</div>") 
        lb_create_vs_load_doc = gr.HTML("<div id=\"cvs_loading\" class=\"border-left-ready substep\">Load the Documents</div>") 
        lb_create_vs_process_doc = gr.HTML("<div id=\"cvs_chunking\" class=\"border-left-ready substep\">Document Processing: Chunking Text for the Language Model</div>") 
        lb_create_vs_embed = gr.HTML("<div id=\"cvs_embedding\" class=\"border-left-ready substep\">Generating Embeddings</div>") 
        lb_create_vs_store = gr.HTML("<div id=\"cvs_storing\" class=\"border-left-ready substep\">Storing them in Chroma</div>") 


        gr.HTML("<div id=\"svs\" class=\"mainstep\"> <span class=\"state ready\"></span>02.serve-vectorstore</div>")
        gr.HTML("<div id=\"svs_logging_vs\" class=\"border-left-ready substep\">Logging the Vector Store as an Artifact</div>") 
        gr.HTML("<div id=\"svs_creating_service\" class=\"border-left-ready substep\">Creating the Inference Service</div>") 


        gr.HTML("<div id=\"sll\" class=\"mainstep\"> <span class=\"state ready\"></span>03.serve-llm</div>")
        gr.HTML("<div id=\"sll_creating_service\" class=\"border-left-ready substep\">Creating the Inference Service</div>") 


        # Debug
        btn_clean = gr.Button("Clean", elem_id="clean", visible=True)
        btn_clean.click(fn=clean, js="location.reload")



    app.launch(server_name="0.0.0.0", server_port=8080, allowed_paths=["image/app-header.png"])
