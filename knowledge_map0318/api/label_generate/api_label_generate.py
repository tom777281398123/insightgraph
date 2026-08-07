"""
实体打标签接口
"""
from flask import Blueprint, request, jsonify, g
from log.log import logging
from tools.label_generate.label_generate import entity_labeling

blueprint_entity_labeling = Blueprint('blueprint_entity_labeling', __name__)


@blueprint_entity_labeling.route("/ekg/entityLabeling", methods=['POST'])
def entity_labeling_api():
    try:
        data = request.json
        logging.info(f"Request data: {str(data)}")

        task_id = data.get("task_id", "")
        task_name = data.get("task_name", "")
        labels = data.get("labels", [])
        entities = data.get("entities", [])

        if not entities:
            return jsonify({"code": 400, "message": "entities is required", "data": []})
        if not labels:
            return jsonify({"code": 400, "message": "labels is required", "data": []})

        tokenizer = g.tokenizer
        model = g.model

        result = entity_labeling(task_id, task_name, labels, entities, tokenizer, model)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in entity_labeling_api: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"Server Error: {str(e)}",
            "data": []
        })
