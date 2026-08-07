# 抽取的三元组格式转换
import uuid
import json

# 原始数据（简化版，实际使用时可以直接加载）
file_path = "tools/knowledge_extraction/knowledge_extraction_result_2025-03-25_09-44-26.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 创建一个实体存储字典
entities = {}
relations = []
relation_set = set()  # 用于去重关系

# 生成 UUID
def generate_uuid():
    return str(uuid.uuid4())

# 处理数据
def process_data(data):
    entity_map = {}  # 用于存储实体名 -> UUID，防止重复创建
    for file_key, triples in data["data"]["extractionIResult"].items():
        for triple in triples:
            head_entity = triple["headEntity"]
            tail_entity = triple["tailEntity"]
            relation_name = triple["relationName"]

            # 处理头实体
            if head_entity["entityName"] not in entity_map:
                entity_id = f"entity_uuid_{generate_uuid()}"
                entities[entity_id] = {
                    "entityId": entity_id,
                    "entityName": head_entity["entityName"],
                    "ontologyId": head_entity["ontologyId"],
                    "attributes": head_entity["attributes"]
                }
                entity_map[head_entity["entityName"]] = entity_id
            else:
                entity_id = entity_map[head_entity["entityName"]]

            # 处理尾实体
            if tail_entity["entityName"] not in entity_map:
                tail_id = f"entity_uuid_{generate_uuid()}"
                entities[tail_id] = {
                    "entityId": tail_id,
                    "entityName": tail_entity["entityName"],
                    "ontologyId": tail_entity["ontologyId"],
                    "attributes": tail_entity["attributes"]
                }
                entity_map[tail_entity["entityName"]] = tail_id
            else:
                tail_id = entity_map[tail_entity["entityName"]]

            # 关系去重
            relation_key = (entity_id, relation_name, tail_id)
            if relation_key not in relation_set:
                relation_set.add(relation_key)
                relations.append({
                    "headEntityId": entity_id,
                    "tailEntityId": tail_id,
                    "relationName": relation_name,
                    "relationId": f"relation_uuid_{generate_uuid()}"
                })

# 执行数据处理
process_data(data)

# 生成输出格式
output = {
    "entities": list(entities.values()),
    "relations": relations
}

# # 输出结果
# print(json.dumps(output, indent=4, ensure_ascii=False))
# 将输出保存为 JSON 文件
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4, ensure_ascii=False)