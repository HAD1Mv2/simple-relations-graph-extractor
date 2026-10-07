import re
import uuid
from collections import defaultdict
from typing import TypedDict

import spacy
from fastcoref import spacy_component
from nltk.tokenize import sent_tokenize
from torch import device as Device
from transformers import pipeline

ENTITY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "myapp://namespace//entity_object")
RELATION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "myapp://namespace//relation_object")

class RelDict(TypedDict):
    entity1_id: str
    entity2_id: str
    relation_type: str
    r_id: str

class EntityDict(TypedDict):
    name: str
    ner_tag: str
    e_id: str

class RelationsExtractor:
    """Class for Relarion Extractor

    Parameters
    ----------
    ner_model_checkpoint : str
        NER model checkpoint.
    rel_model_checkpoint : str
        Relation Classification model checkpoint.
    device : Device
        Device used to load the model components.
    """
    def __init__(self, ner_model_checkpoint: str, rel_model_checkpoint: str, device: Device):
        self.ner_pipe = pipeline("token-classification", model=ner_model_checkpoint, aggregation_strategy="max", device=device)
        self.relation_classifier = pipeline("text-classification", model=rel_model_checkpoint, device=device)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.spacy_nlp.add_pipe("fastcoref")

        self.class_group = {'ORG:located at(e1,e2)': ['org:stateorprovince_of_headquarters(e1,e2)', 'org:city_of_headquarters(e1,e2)', 'org:country_of_headquarters(e1,e2)'], 
                            'ORG:located at(e2,e1)': ['org:stateorprovince_of_headquarters(e2,e1)', 'org:city_of_headquarters(e2,e1)', 'org:country_of_headquarters(e2,e1)'], 
                            'PER:located at(e1,e2)': ['per:countries_of_residence(e1,e2)', 'per:cities_of_residence(e1,e2)', 'per:stateorprovinces_of_residence(e1,e2)'], 
                            'PER:located at(e2,e1)': ['per:countries_of_residence(e2,e1)', 'per:cities_of_residence(e2,e1)', 'per:stateorprovinces_of_residence(e2,e1)'], 
                            'PER:member of(e1,e2)': ['org:founded_by(e1,e2)', 'per:employee_of(e1,e2)', 'org:top_members/employees(e1,e2)'], 
                            'PER:member of(e2,e1)': ['org:founded_by(e2,e1)', 'per:employee_of(e2,e1)', 'org:top_members/employees(e2,e1)']}

    def _clean_text(self, input_text: str) -> str:
        """Simple cleaning text function.

        Parameters
        ----------
        input_text : str
            Text to clean. 

        Returns
        -------
        str
            Cleaned text.
        """
        
        input_text = re.sub("  ", " ", input_text)
        input_text = re.sub(r"\( ", "(", input_text)
        input_text = re.sub(r" \)", ")", input_text)
        input_text = input_text.strip()
        return input_text

    def _filter_relation_prediction(self, ner_tag_entity1: str, ner_tag_entity2: str, relation_prediction: str) -> tuple[str, str]:
        """Filter and grouping relation class 

        Parameters
        ----------
        ner_tag_entity1 : str
            NER tag for enitity 1 (e1), e.g., PER, ORG, LOC.
        ner_tag_entity2 : str
            NER tag for enitity 2 (e2).
        relation_prediction : str
            Relation class, output from relation classificatiojn model, e.g., org:founded_by(e1,e2), per:cities_of_residence(e1,e2), org:stateorprovince_of_headquarters(e1,e2)
            check kbp37 dataset for full class category.

        Returns
        -------
        tuple[str, str]
            Filtered class and the direction of relation
        """
        direction = "None"
        if relation_prediction in self.class_group['ORG:located at(e1,e2)']:
            if ner_tag_entity1 == 'ORG' and ner_tag_entity2 == 'LOC':
                relation_prediction = 'located at'
                direction = "right"
            else:
                relation_prediction = "Other"
        elif relation_prediction in self.class_group['ORG:located at(e2,e1)']:
            if ner_tag_entity1 == 'LOC' and ner_tag_entity2 == 'ORG':
                relation_prediction = 'located at'
                direction = "left"
            else:
                relation_prediction = "Other"
        elif relation_prediction in self.class_group['PER:located at(e1,e2)']:
            if ner_tag_entity1 == 'PER' and ner_tag_entity2 == 'LOC':
                relation_prediction = 'located at'
                direction = "right"
            else:
                relation_prediction = "Other"
        elif relation_prediction in self.class_group['PER:located at(e2,e1)']:
            if ner_tag_entity1 == 'LOC' and ner_tag_entity2 == 'PER':
                relation_prediction = 'located at'
                direction = "left"
            else:
                relation_prediction = "Other"
        elif relation_prediction in self.class_group['PER:member of(e1,e2)']:
            if ner_tag_entity1 == 'PER' and ner_tag_entity2 == 'ORG':
                relation_prediction = 'member of'
                direction = "right"
            else:
                relation_prediction = "Other"
        elif relation_prediction in self.class_group['PER:member of(e2,e1)']:
            if ner_tag_entity1 == 'ORG' and ner_tag_entity2 == 'PER':
                relation_prediction = 'member of'
                direction = "left"
            else:
                relation_prediction = "Other"
        elif relation_prediction=='per:origin(e1,e2)':
            if ner_tag_entity1 == 'PER' and ner_tag_entity2 == 'LOC':
                relation_prediction = "origin"  
                direction = "right"
            else:
                relation_prediction = "Other"
        elif relation_prediction=='per:origin(e2,e1)':
            if ner_tag_entity1 == 'LOC' and ner_tag_entity2 == 'PER':
                relation_prediction = "origin"  
                direction = "left"
            else:
                relation_prediction = "Other"       
        else:
            relation_prediction = 'Other'

        return relation_prediction, direction

    def _register_entity(self, entity: dict, entity_id_registry: defaultdict, entities: list):

        # if the entity haven't registered, then register
        if not entity_id_registry[entity["e_id"]]:
            entity_id_registry[entity["e_id"]] = True
            entities.append(entity)

    def _register_relation(self, relation: dict, relation_id_registry: defaultdict, relations: list):

        # if the relations haven't registered, then register
        if not relation_id_registry[relation["r_id"]]:
            relation_id_registry[relation["r_id"]] = True
            relations.append(relation)

    def extract_from_sentence(self, input_sentence: str) -> tuple[list[EntityDict], list[RelDict]]:
        """Extract relation data from a sentence.

        Parameters
        ----------
        input_sentence : str
            Input sentence.

        Returns
        -------
        list[RelDict]
            Lit containing relation data.
        """
        # get entities
        ner_result = self.ner_pipe(self._clean_text(input_sentence))

        len_ner_result = len(ner_result)
        entity_nodes = []
        relation_prediction_results = []

        for i in range(len_ner_result-1):
            for j in range(i+1, len_ner_result):
                tagged_input_text = input_sentence[:ner_result[i]['start']] + '<e1> ' + input_sentence[ner_result[i]['start']:ner_result[i]['end']] + ' </e1>' + input_sentence[ner_result[i]['end']:ner_result[j]['start']] + '<e2> ' + input_sentence[ner_result[j]['start']:ner_result[j]['end']] + ' </e2>' +  input_sentence[ner_result[j]['end']:]
                relation_prediction = self.relation_classifier(tagged_input_text)
                relation_prediction, direction = self._filter_relation_prediction(ner_result[i]['entity_group'], ner_result[j]['entity_group'], relation_prediction[0]['label'])
                if relation_prediction != "Other":
                    entity_front = {"name": input_sentence[ner_result[i]['start']:ner_result[i]['end']], "ner_tag":ner_result[i]['entity_group']}
                    entity_back = {"name": input_sentence[ner_result[j]['start']:ner_result[j]['end']], "ner_tag":ner_result[j]['entity_group']}

                    # set entity id
                    entity_front["e_id"] = str(uuid.uuid5(ENTITY_NAMESPACE, entity_front["name"].lower()+"_"+entity_front["ner_tag"]))
                    entity_back["e_id"] = str(uuid.uuid5(ENTITY_NAMESPACE, entity_back["name"].lower()+"_"+entity_back["ner_tag"]))

                    # register entity id
                    entity_nodes.append(entity_front) 
                    entity_nodes.append(entity_back) 

                    if direction == "right":
                        relation = {'entity1_id': entity_front["e_id"],
                                    'entity2_id': entity_back["e_id"],
                                    'relation_type': relation_prediction}
                    else:
                        relation = {'entity1_id': entity_back["e_id"],
                                    'entity2_id': entity_front["e_id"],
                                    'relation_type': relation_prediction}

                    relation["r_id"] = str(uuid.uuid5(RELATION_NAMESPACE, relation['entity1_id']+"_"+relation['entity2_id']+"_"+relation['relation_type']))
                    relation_prediction_results.append(relation)

        return entity_nodes, relation_prediction_results


    def extract_from_article(self, input_article: str) -> tuple[list[EntityDict],list[RelDict]]:
        """Extract relation data from an article text

        Parameters
        ----------
        input_article : str
            Text article.

        Returns
        -------
        list[RelDict]
            List containing relation data.
        """

        # create entity registry for each article (not for global knowledge, since this is simple graph extraction)
        entity_id_registry = defaultdict(bool)
        entities = []

        relation_id_registry = defaultdict(bool)
        relations = []        

        doc = self.spacy_nlp(input_article, component_cfg={"fastcoref":{'resolve_text': True}})
        coref_article = doc._.resolved_text
        input_sentences = sent_tokenize(coref_article)

        for sentence in input_sentences:
            entity_nodes, relations_result = self.extract_from_sentence(sentence)

            # register entities
            for entity in entity_nodes:
                self._register_entity(entity, entity_id_registry, entities)

            # register relations
            for relation in relations_result:
                self._register_relation(relation, relation_id_registry, relations)

        return entities, relations