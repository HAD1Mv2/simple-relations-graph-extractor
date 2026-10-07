import json

import requests
import streamlit as st
from neo4j_viz import Node, Relationship, VisualizationGraph
from neo4j_viz.options import Renderer

if 'headers' not in st.session_state:
    st.session_state['headers'] = {'app-key':'12345XOXO'}

if 'blank-state' not in st.session_state:
    st.session_state['blank-state'] = True
    st.session_state['show-hide-json-state'] = False

if 'relations-data' not in st.session_state:
    st.session_state['relations-data'] = {}

@st.cache_data
def fetch_relations(input_article):

    input_json = {'article': input_article}
    relations_data= requests.post('http://127.0.0.1:1001/get_relations', json=input_json, headers = st.session_state.headers)

    return relations_data

@st.cache_data
def create_relationship_graph_(api_result):

    entities = api_result["entities_data"]
    relations = api_result["relations_data"]

    nodes = [Node(id=member["e_id"], caption=member["name"], properties={"ner_tag": member["ner_tag"]}) for member in entities]
    relationships = [Relationship(source=element["entity1_id"], target=element["entity2_id"], caption=element["relation_type"]) for element in relations]

    vg = VisualizationGraph(nodes=nodes, relationships=relationships)

    # color the graph nodes based on entity type
    vg.color_nodes(property="ner_tag")
    vg.color_relationships(field="caption")

    visualize_status = "done"

    return vg, visualize_status 

st.markdown("""
# Simple Relation Graph Extraction
This app extract knowledge graph from an article
""")

st.header("Enter content of the article")

with open("example/input_article.txt", 'r', encoding='utf-8') as f:
    example_article = f.read()

article_content = st.text_area("article content", example_article, height=250)

execute_button = st.button('Extract relations data')

if execute_button:
    # try:
    relations_data = fetch_relations(article_content)
    if relations_data.status_code == 200:
        if relations_data.json()['status'] == 1:
            graph, visualize_status = create_relationship_graph_(relations_data.json())
            # Save the visualization to a file
            with open("neo4j_graph.html", "w") as f:
                f.write(graph.render(renderer=Renderer.CANVAS).data)
            if visualize_status == 'done':
                st.session_state['blank-state'] = False
                st.session_state['relations-data'] = relations_data.json()
            else:
                st.text("Error!!!, Cannot visualize the graph")
                st.session_state['blank-state'] = True
        else:
            st.text("Error!!!, Failed to extract relations data")
            st.session_state['blank-state'] = True
    else:
        st.text("Failed to extract relations!!!")
        st.session_state['blank-state'] = True

if not st.session_state['blank-state']:

    with open("neo4j_graph.html", 'r', encoding='utf-8') as htmlfile :
        source_code = htmlfile.read() 

    with st.container():
        st.iframe(source_code)
        show_hide_json_button = st.button('Show/Hide raw data')
    if show_hide_json_button:
        st.session_state['show-hide-json-state'] = not st.session_state['show-hide-json-state']

if st.session_state['show-hide-json-state']:
    st.json(st.session_state['relations-data'])
    object_to_download = json.dumps(st.session_state['relations-data'])    
    st.download_button(
        label="Download JSON",
        data=object_to_download,
        file_name='relations_data.json',
        mime='application/json',
    )