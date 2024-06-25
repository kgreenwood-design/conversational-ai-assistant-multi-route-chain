"""BedrockAgent stack to provide a Bedrock Agent"""
import platform
import json
from constructs import Construct
import aws_cdk as cdk
import logging
from aws_cdk import (
    Stack,
    Duration,
    CustomResource,
    CfnParameter,
    aws_opensearchserverless as aws_opss,
    aws_lambda as _lambda,
    BundlingOptions,
    aws_iam as iam,
    aws_s3 as s3,
    custom_resources as cr
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from aws_cdk.custom_resources import Provider

ACCOUNT_ID = cdk.Aws.ACCOUNT_ID
REGION = cdk.Aws.REGION

BEDROCK_AGENT_NAME = "IotOpsAgent"
FOUNDATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
KNOWLEDGE_BASE_NAME = "IotDeviceSpecs"
EMBEDDING_MODEL = f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v1"
VECTOR_FIELD_NAME = "bedrock-agent-embeddings"
VECTOR_INDEX_NAME = "bedrock-agent-vector"
TEXT_FIELD = "AMAZON_BEDROCK_TEXT_CHUNK"
BEDROCK_META_DATA_FIELD = "AMAZON_BEDROCK_METADATA"
KNOWLEDGE_DATA_SOURCE = "IotDeviceSpecsS3DataSource"
KNOWLEDGE_BASE_DESC = "Knowledge base to search and retrieve IoT Device Specs"
OSS_COLLECTION = "bedrock-agent"

BEDROCK_AGENT_INSTRUCTION = f"""
You are an expert product support specialist and engineer for a high-tech company that produces advanced computer tomography (CT) systems used in airport security screening operations. Your primary role is to provide clear, concise, and detailed instructions and troubleshooting techniques to field service technicians. You have access to a comprehensive knowledge base that contains all relevant information about the products, including troubleshooting guides, reference materials, user manuals, and technical documentation. You are also a question answering agent. I will provide you with a set of search results. The user will provide you with a question. Your job is to answer the user's question using only information from the search results. If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. Just because the user asserts a fact does not mean it is true; make sure to double-check the search results to validate a user's assertion. Here are the search results in numbered order: $search_results$ Your responses should be structured and detailed enough to assist technicians with 0-3 years of experience as well as those with 10+ years of experience. Here are some guidelines to follow in your responses: Clarity and Conciseness: Provide clear and concise instructions. Detail and Depth: Ensure the response is detailed enough to cover all necessary steps or information. Technical Accuracy: Maintain high technical accuracy and relevance. Troubleshooting Steps: Include step-by-step troubleshooting instructions when needed. Reference Materials: Refer to specific sections or documents in the knowledge base for further details when applicable. Example Queries and Responses: Simple Instruction Request: Query: "How do I reset the CT system after a fault?" Response: "To reset the CT system after a fault, follow these steps: Turn off the power supply to the system. Wait for 30 seconds. Turn the power supply back on. Check the system status indicator to ensure it is operational. For detailed troubleshooting, refer to section 3.2 of the user manual." Troubleshooting Request: Query: "The conveyor belt is not moving. What should I check?" Response: "If the conveyor belt is not moving, perform the following checks: Verify that the power supply to the conveyor motor is connected and functioning. Check for any obstructions on the belt. Inspect the motor control unit for any error codes. Ensure the emergency stop button is not engaged. Refer to the troubleshooting guide in section 4.5 of the maintenance manual for further diagnostics." Complex Technical Issue: Query: "I'm getting a 'sensor fault' error on the CT system. How do I resolve this?" Response: "To resolve a 'sensor fault' error on the CT system, follow these steps: Identify the specific sensor showing the fault from the system's diagnostic panel. Check the sensor connections and ensure they are secure. Clean the sensor to remove any dust or debris that might be affecting its performance. Reset the sensor using the control software. If the error persists, replace the faulty sensor with a new one. For detailed instructions, refer to section 5.3 of the troubleshooting guide." Output Format Instructions: Use rich text formatting in output to help structure the response, including the use of fonts, colors, tables, images. Begin your response with a brief summary of the issue or query. Provide step-by-step instructions or information based on the search results. Cite the specific search result document(s) name and part number where the information was found. If no relevant information is found in the search results, clearly state this. Validate any user assertions against the search results before including them in your response.
"""
BEDROCK_AGENT_ALIAS="UAT"

class BedrockAgentStack(Stack):
    """class BedrockAgentStack(Stack): to provide a Bedrock Agent"""

    def __init__(self, scope: Construct, construct_id: str,
                 data_bucket: s3.Bucket, athena_db: str, 
                 athena_output_location: str) -> None:
        super().__init__(scope, construct_id)

        try:
            logger.info("Initializing BedrockAgentStack")
            self.check_bedrock_availability()
            custom_res_role = self.create_custom_resource_role()
            bedrock_agent_role = self.create_bedrock_agent_role(data_bucket)
            bedrock_kb_role = self.create_bedrock_kb_role()
            agent_id, agent_alias_id = self.create_bedrock_agent(custom_res_role, bedrock_agent_role, bedrock_kb_role)
            self.bedrock_agent_id = agent_id
            self.bedrock_agent_alias = agent_alias_id
            logger.info("BedrockAgentStack initialization completed successfully")
        except Exception as e:
            logger.error(f"Error in BedrockAgentStack initialization: {str(e)}")
            raise

    def check_bedrock_availability(self):
        try:
            bedrock_client = boto3.client('bedrock')
            bedrock_client.list_foundation_models()
            logger.info("Bedrock service is available")
        except Exception as e:
            logger.error(f"Bedrock service is not available: {str(e)}")
            raise

    def create_custom_resource_role(self):
        custom_res_role = iam.Role(
            self, "CustomResourceRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name="CustomResourceRole"
        )
        custom_res_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AdministratorAccess"
            )
        )
        logger.info("Custom resource role created successfully")
        return custom_res_role

    def create_bedrock_agent_role(self, data_bucket):
        bedrock_agent_role = iam.Role(
            self, "BedrockAgentRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            role_name=f"AmazonBedrockExecutionRoleForAgents_{BEDROCK_AGENT_NAME}"
        )
        data_bucket.grant_read_write(bedrock_agent_role)
        self.attach_bedrock_agent_policies(bedrock_agent_role)
        logger.info("Bedrock agent role created successfully")
        return bedrock_agent_role

    def attach_bedrock_agent_policies(self, role):
        policies = [
            ("BedrockAgentLambdaPolicy", ["lambda:InvokeFunction"]),
            ("BedrockAgentS3Policy", ["s3:GetObject"]),
            ("BedrockAgentModelPolicy", ["bedrock:*"])
        ]
        for policy_name, actions in policies:
            policy = iam.Policy(
                self, policy_name,
                policy_name=policy_name,
                statements=[
                    iam.PolicyStatement(
                        actions=actions,
                        resources=["*"]
                    )
                ]
            )
            role.attach_inline_policy(policy)

    def create_bedrock_kb_role(self):
        bedrock_kb_role = iam.Role(
            self, "BedrockKbRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            role_name=f"AmazonBedrockExecutionRoleForKnowledgeBase_{BEDROCK_AGENT_NAME}",
        )
        logger.info("Bedrock knowledge base role created successfully")
        return bedrock_kb_role

    def create_bedrock_agent(self, custom_res_role, bedrock_agent_role, bedrock_kb_role):
        try:
            agent_res = cr.AwsCustomResource(
                scope=self,
                id='BedrockAgent',
                role=custom_res_role,
                on_create=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="CreateAgent",
                    parameters={
                        "agentName": BEDROCK_AGENT_NAME,
                        "agentResourceRoleArn": bedrock_agent_role.role_arn,
                        "foundationModel": FOUNDATION_MODEL,
                        "instruction": BEDROCK_AGENT_INSTRUCTION
                    },
                    physical_resource_id=cr.PhysicalResourceId.from_response("agentId"),
                    output_paths=["agentId"]
                ),
                on_delete=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="DeleteAgent",
                    parameters={
                        "agentId": cr.PhysicalResourceIdReference(),
                        "skipResourceInUseCheck": True
                    }
                ),
            )
            agent_id = agent_res.get_response_field("agentId")
            
            agent_alias_res = cr.AwsCustomResource(
                scope=self,
                id='BedrockAgentAlias',
                role=custom_res_role,
                on_create=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="CreateAgentAlias",
                    parameters={
                        "agentId": agent_id,
                        "agentAliasName": BEDROCK_AGENT_ALIAS
                    },
                    physical_resource_id=cr.PhysicalResourceId.of("id"),
                    output_paths=["agentAlias.agentAliasId"]
                )
            )
            agent_alias_id = agent_alias_res.get_response_field("agentAlias.agentAliasId")
            
            logger.info(f"Bedrock agent created successfully. Agent ID: {agent_id}, Alias ID: {agent_alias_id}")
            return agent_id, agent_alias_id
        except Exception as e:
            logger.error(f"Error creating Bedrock agent: {str(e)}")
            raise

    def check_bedrock_availability(self):
        try:
            bedrock_client = boto3.client('bedrock')
            bedrock_client.list_foundation_models()
            logger.info("Bedrock service is available")
        except Exception as e:
            logger.error(f"Bedrock service is not available: {str(e)}")
            raise

    def create_custom_resource_role(self):
        custom_res_role = iam.Role(
            self, "CustomResourceRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            role_name="CustomResourceRole"
        )
        custom_res_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AdministratorAccess"
            )
        )
        logger.info("Custom resource role created successfully")
        return custom_res_role

    def create_bedrock_agent_role(self, data_bucket):
        bedrock_agent_role = iam.Role(
            self, "BedrockAgentRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            role_name=f"AmazonBedrockExecutionRoleForAgents_{BEDROCK_AGENT_NAME}"
        )
        data_bucket.grant_read_write(bedrock_agent_role)
        self.attach_bedrock_agent_policies(bedrock_agent_role)
        logger.info("Bedrock agent role created successfully")
        return bedrock_agent_role

    def attach_bedrock_agent_policies(self, role):
        policies = [
            ("BedrockAgentLambdaPolicy", ["lambda:InvokeFunction"]),
            ("BedrockAgentS3Policy", ["s3:GetObject"]),
            ("BedrockAgentModelPolicy", ["bedrock:*"])
        ]
        for policy_name, actions in policies:
            policy = iam.Policy(
                self, policy_name,
                policy_name=policy_name,
                statements=[
                    iam.PolicyStatement(
                        actions=actions,
                        resources=["*"]
                    )
                ]
            )
            role.attach_inline_policy(policy)

    def create_bedrock_kb_role(self):
        bedrock_kb_role = iam.Role(
            self, "BedrockKbRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            role_name=f"AmazonBedrockExecutionRoleForKnowledgeBase_{BEDROCK_AGENT_NAME}",
        )
        logger.info("Bedrock knowledge base role created successfully")
        return bedrock_kb_role

    def create_bedrock_agent(self, custom_res_role, bedrock_agent_role, bedrock_kb_role):
        try:
            agent_res = cr.AwsCustomResource(
                scope=self,
                id='BedrockAgent',
                role=custom_res_role,
                on_create=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="CreateAgent",
                    parameters={
                        "agentName": BEDROCK_AGENT_NAME,
                        "agentResourceRoleArn": bedrock_agent_role.role_arn,
                        "foundationModel": FOUNDATION_MODEL,
                        "instruction": BEDROCK_AGENT_INSTRUCTION
                    },
                    physical_resource_id=cr.PhysicalResourceId.from_response("agentId"),
                    output_paths=["agentId"]
                ),
                on_delete=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="DeleteAgent",
                    parameters={
                        "agentId": cr.PhysicalResourceIdReference(),
                        "skipResourceInUseCheck": True
                    }
                ),
            )
            agent_id = agent_res.get_response_field("agentId")
            
            # Create agent alias
            agent_alias_res = cr.AwsCustomResource(
                scope=self,
                id='BedrockAgentAlias',
                role=custom_res_role,
                on_create=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="CreateAgentAlias",
                    parameters={
                        "agentId": agent_id,
                        "agentAliasName": BEDROCK_AGENT_ALIAS
                    },
                    physical_resource_id=cr.PhysicalResourceId.of("id"),
                    output_paths=["agentAlias.agentAliasId"]
                )
            )
            agent_alias_id = agent_alias_res.get_response_field("agentAlias.agentAliasId")
            
            logger.info(f"Bedrock agent created successfully. Agent ID: {agent_id}, Alias ID: {agent_alias_id}")
            return agent_id, agent_alias_id
        except Exception as e:
            logger.error(f"Error creating Bedrock agent: {str(e)}")
            raise
        bedrock_kb_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AdministratorAccess"
            )        
        )

        # create aoss collection
        network_security_policy = json.dumps([{
            "Rules": [
                {
                    "Resource": [
                        f"collection/{OSS_COLLECTION}"
                    ],
                    "ResourceType": "dashboard"
                },
                {
                    "Resource": [
                        f"collection/{OSS_COLLECTION}"
                    ],
                    "ResourceType": "collection"
                }
            ],
            "AllowFromPublic": True,
        }], indent=2)

        cfn_network_security_policy = aws_opss.CfnSecurityPolicy(self, "NetworkSecurityPolicy",
                                                                 policy=network_security_policy,
                                                                 name=f"{OSS_COLLECTION}-security-policy",
                                                                 type="network"
                                                                 )
        encryption_security_policy = json.dumps({
            "Rules": [
                {
                    "Resource": [
                        f"collection/{OSS_COLLECTION}"
                    ],
                    "ResourceType": "collection"
                }
            ],
            "AWSOwnedKey": True
        }, indent=2)

        cfn_encryption_security_policy = aws_opss.CfnSecurityPolicy(self, "EncryptionSecurityPolicy",
                                                                    policy=encryption_security_policy,
                                                                    name=f"{OSS_COLLECTION}-security-policy",
                                                                    type="encryption"
                                                                    )
        cfn_collection = aws_opss.CfnCollection(self, "OpssSearchCollection",
                                                name=OSS_COLLECTION,
                                                description="Collection to be used for search using OpenSearch Serverless vector search",
                                                type="VECTORSEARCH"
                                                )
        cfn_collection.add_dependency(cfn_network_security_policy)
        cfn_collection.add_dependency(cfn_encryption_security_policy)

        data_access_policy = json.dumps([
            {
                "Rules": [
                    {
                        "Resource": [
                            f"collection/{OSS_COLLECTION}"
                        ],
                        "Permission": [
                            "aoss:CreateCollectionItems",
                            "aoss:DeleteCollectionItems",
                            "aoss:UpdateCollectionItems",
                            "aoss:DescribeCollectionItems"
                        ],
                        "ResourceType": "collection"
                    },
                    {
                        "Resource": [
                            f"index/{OSS_COLLECTION}/*"
                        ],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:DeleteIndex",
                            "aoss:UpdateIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument"
                        ],
                        "ResourceType": "index"
                    }
                ],
                "Principal": [
                    f"{custom_res_role.role_arn}",
                    f"{bedrock_agent_role.role_arn}",
                    f"{bedrock_kb_role.role_arn}",
                ],
                "Description": "data-access-rule"
            }
        ], indent=2)

        data_access_policy_name = f"{OSS_COLLECTION}-access-policy"
        assert len(data_access_policy_name) <= 32

        data_access_policy = aws_opss.CfnAccessPolicy(self, "OpssDataAccessPolicy",
                                 name=data_access_policy_name,
                                 description="Policy for data access",
                                 policy=data_access_policy,
                                 type="data"
                                 )
        data_access_policy.add_dependency(cfn_collection)
        
        # create aoss index
        platform_mapping = {
            "x86_64": _lambda.Architecture.X86_64,
            "arm64": _lambda.Architecture.ARM_64
        }
        architecture = platform_mapping[platform.uname().machine]
        index_res_lambda = _lambda.Function(
            self, "OpenSearchIndexLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            code=_lambda.Code.from_asset('bedrock_agent_implementation/custom_resource/aoss',
                                         bundling=BundlingOptions(
                                             image=_lambda.Runtime.PYTHON_3_11.bundling_image,
                                             command=['bash',
                                                      '-c',
                                                      'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'])),
            handler='index.on_event',
            architecture=architecture,
            timeout=Duration.seconds(900),
            role=custom_res_role,
            environment={
                "VECTOR_FIELD_NAME": VECTOR_FIELD_NAME,
                "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
                "TEXT_FIELD": TEXT_FIELD,
                "BEDROCK_META_DATA_FIELD": BEDROCK_META_DATA_FIELD,
            }
        )
        index_res_provider = Provider(self, "OpenSearchIndexResProvider",
                                       on_event_handler=index_res_lambda,
                                       )

        index_res = CustomResource(self, "IndexOpenSearch", service_token=index_res_provider.service_token,
                                    properties={
                                        "oss_endpoint": cfn_collection.attr_collection_endpoint,
                                    })
        index_res.node.add_dependency(data_access_policy)

        #create knowledge base with the opensearch index
        bedrock_kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["aoss:APIAccessAll"],
                resources=[cfn_collection.attr_arn]
            )
        )
        bedrock_kb_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[EMBEDDING_MODEL]
            )
        )

        kb_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockKowledgeBase',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="CreateKnowledgeBase",
                parameters={
                    "knowledgeBaseConfiguration": {
                        "type": "VECTOR",
                        "vectorKnowledgeBaseConfiguration": {
                            "embeddingModelArn": EMBEDDING_MODEL
                        }
                    },
                    "name": KNOWLEDGE_BASE_NAME,
                    "roleArn": bedrock_kb_role.role_arn,
                    "storageConfiguration": {
                        "type": "OPENSEARCH_SERVERLESS",
                        "opensearchServerlessConfiguration": { 
                            "collectionArn": cfn_collection.attr_arn,
                            "fieldMapping": { 
                                "metadataField": BEDROCK_META_DATA_FIELD,
                                "textField": TEXT_FIELD,
                                "vectorField": VECTOR_FIELD_NAME
                                },
                            "vectorIndexName": VECTOR_INDEX_NAME
                        }
                    },
                },
                physical_resource_id=cr.PhysicalResourceId.from_response("knowledgeBase.knowledgeBaseId"),
                output_paths=["knowledgeBase.knowledgeBaseId"]
            ),
            on_delete=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="DeleteKnowledgeBase",
                parameters={
                    "knowledgeBaseId": cr.PhysicalResourceIdReference(),
                }
            )
        )
        kb_res.node.add_dependency(index_res)
        kb_res.node.add_dependency(bedrock_kb_role)
        knowledgebase_id = kb_res.get_response_field("knowledgeBase.knowledgeBaseId")
        
        # create data source
        data_source_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockDataSource',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="CreateDataSource",
                parameters={
                    "knowledgeBaseId": knowledgebase_id,
                    "name": KNOWLEDGE_DATA_SOURCE,
                    "dataDeletionPolicy": "RETAIN",
                    "dataSourceConfiguration": {
                        "type": "S3",
                        "s3Configuration": {
                            "bucketArn": data_bucket.bucket_arn,
                            "inclusionPrefixes": ["iot_device_info/"]
                        }
                    }
                },
                physical_resource_id=cr.PhysicalResourceId.from_response("dataSource.dataSourceId"),
                output_paths=["dataSource.dataSourceId"]
            )
        )  
        data_source_res.node.add_dependency(kb_res)
        datasource_id = data_source_res.get_response_field("dataSource.dataSourceId")
        # start datasource ingestion job
        ingestion_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockIngestion',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="StartIngestionJob",
                parameters={
                    "knowledgeBaseId": knowledgebase_id,
                    "dataSourceId": datasource_id,
                },
                physical_resource_id=cr.PhysicalResourceId.of("id"),
            )
        )
        ingestion_res.node.add_dependency(data_source_res)

        # create a bedrock agent
        try:
            agent_res = cr.AwsCustomResource(
                scope=self,
                id='BedrockAgent',
                role=custom_res_role,
                on_create=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="CreateAgent",
                    parameters={
                        "agentName": BEDROCK_AGENT_NAME,
                        "agentResourceRoleArn": bedrock_agent_role.role_arn,
                        "foundationModel": FOUNDATION_MODEL,
                        "instruction": BEDROCK_AGENT_INSTRUCTION
                    },
                    physical_resource_id=cr.PhysicalResourceId.from_response("agentId"),
                    output_paths=["agentId"]
                ),
                on_delete=cr.AwsSdkCall(
                    service="@aws-sdk/client-bedrock-agent",
                    action="DeleteAgent",
                    parameters={
                        "agentId": cr.PhysicalResourceIdReference(),
                        "skipResourceInUseCheck": True
                    }
                ),
            )
            logger.info("Bedrock agent created successfully")
        except Exception as e:
            logger.error(f"Error creating Bedrock agent: {str(e)}")
            raise
        agent_res.node.add_dependency(bedrock_agent_role)
        agent_res.node.add_dependency(bedrock_agent_lambda_policy)
        agent_res.node.add_dependency(bedrock_agent_s3_policy)
        agent_res.node.add_dependency(bedrock_agent_model_policy)
        agent_res.node.add_dependency(kb_res)

        agent_id = agent_res.get_response_field("agentId")

        # update agent to associate with the knowledge base
        associate_agent_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockAssociateKb',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="AssociateAgentKnowledgeBase",
                parameters={
                    "agentId": agent_id,
                    "agentVersion": "DRAFT",
                    "knowledgeBaseId": knowledgebase_id,
                    "knowldeBaseState": "ENABLED",
                    "description": KNOWLEDGE_BASE_DESC
                },
                physical_resource_id=cr.PhysicalResourceId.of("id"),
            ) 
        )  
        associate_agent_res.node.add_dependency(data_source_res)
        associate_agent_res.node.add_dependency(agent_res)

        # action 1 is the device metrics lambda
        action_1_lambda = _lambda.Function(
            self, "DeviceMetricsLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            code=_lambda.Code.from_asset(
                'bedrock_agent_implementation/action_groups/check_device_metrics_query'),
            handler='lambda_function.lambda_handler',
            timeout=cdk.Duration.seconds(300),
            environment={
                'ATHENA_DATABASE': athena_db,
                'ATHENA_OUTPUT_LOCATION': athena_output_location
            },
        )
        action_1_lambda.add_permission(
            "AllowBedrockAgent",
            principal=iam.ServicePrincipal("bedrock.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_account=ACCOUNT_ID,
            source_arn=f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:agent/{agent_id}"
        )
        
        action_1_lambda.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonAthenaFullAccess")
        )
        data_bucket.grant_read_write(action_1_lambda.role)

        # action 2 is the device action lambda 
        sender = CfnParameter(self, "sender", type="String",
                              description="The sender's email for SES email notification")
        recipient = CfnParameter(self, "recipient", type="String",
                              description="The recipient's email for SES email notification")
        
        action_2_lambda = _lambda.Function(
            self, "DeviceActionLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            code=_lambda.Code.from_asset(
                'bedrock_agent_implementation/action_groups/action_on_device'),
            handler='lambda_function.lambda_handler',
            timeout=cdk.Duration.seconds(300),
            environment={
                'SENDER': sender.value_as_string,
                'RECIPIENT': recipient.value_as_string
            },
        )
        action_2_lambda.add_permission(
            "AllowBedrockAgent",
            principal=iam.ServicePrincipal("bedrock.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_account=ACCOUNT_ID,
            source_arn=f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:agent/{agent_id}"
        )

        action_2_lambda.role.add_to_policy(
            iam.PolicyStatement(
                actions=["ses:SendEmail"],
                resources=["*"]
            )
        )

        agent_action_group_res_1 = cr.AwsCustomResource(
            scope=self,
            id='BedrockAgentActionGroup1',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="CreateAgentActionGroup",
                parameters={
                    "agentId": agent_id,
                    "agentVersion": "DRAFT",
                    "actionGroupExecutor": {
                        "lambda": action_1_lambda.function_arn,
                    },
                    "actionGroupName": "CheckDeviceMetricsActionGroup",
                    "actionGroupState": "ENABLED",
                    "apiSchema": {
                        "s3": {
                            "s3BucketName": data_bucket.bucket_name,
                            "s3ObjectKey": f"open_api_schema/check_device_metrics.json"
                        }
                    }
                },
                physical_resource_id=cr.PhysicalResourceId.from_response("agentActionGroup.actionGroupId"),
                output_paths=["agentActionGroup.actionGroupId"]
            )
        ) 

        agent_action_group_res_1.node.add_dependency(associate_agent_res)
        agent_action_group_res_1.node.add_dependency(action_1_lambda)

        agent_action_group_res_2 = cr.AwsCustomResource(
            scope=self,
            id='BedrockAgentActionGroup2',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="CreateAgentActionGroup",
                parameters={
                    "agentId": agent_id,
                    "agentVersion": "DRAFT",
                    "actionGroupExecutor": {
                        "lambda": action_2_lambda.function_arn,
                    },
                    "actionGroupName": "ActionOnDeviceActionGroup",
                    "actionGroupState": "ENABLED",
                    "apiSchema": {
                        "s3": {
                            "s3BucketName": data_bucket.bucket_name,
                            "s3ObjectKey": f"open_api_schema/action_on_device.json"
                        }
                    }
                },
                physical_resource_id=cr.PhysicalResourceId.from_response("agentActionGroup.actionGroupId"),
                output_paths=["agentActionGroup.actionGroupId"]
            )
        ) 

        agent_action_group_res_2.node.add_dependency(associate_agent_res)
        agent_action_group_res_2.node.add_dependency(action_2_lambda)

        #prepare agent
        prepare_agent_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockPrepareAgent',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="PrepareAgent",
                parameters={
                    "agentId": agent_id
                },
                physical_resource_id=cr.PhysicalResourceId.of("id"),
                output_paths=["agentStatus"]
            )
        )

        prepare_agent_res.node.add_dependency(agent_action_group_res_1)
        prepare_agent_res.node.add_dependency(agent_action_group_res_2)

        #create agent alias
        agent_alias_res = cr.AwsCustomResource(
            scope=self,
            id='BedrockAgentAlias',
            role=custom_res_role,
            on_create=cr.AwsSdkCall(
                service="@aws-sdk/client-bedrock-agent",
                action="CreateAgentAlias",
                parameters={
                    "agentId": agent_id,
                    "agentAliasName": BEDROCK_AGENT_ALIAS
                },
                physical_resource_id=cr.PhysicalResourceId.of("id"),
                output_paths=["agentAlias.agentAliasId"]
            )
        )
        agent_alias_id = agent_alias_res.get_response_field("agentAlias.agentAliasId")
        agent_alias_res.node.add_dependency(prepare_agent_res)

        self.bedrock_agent_id = agent_id
        self.bedrock_agent_alias = agent_alias_id

        logger.info("BedrockAgentStack initialization completed successfully")
    except Exception as e:
        logger.error(f"Error in BedrockAgentStack initialization: {str(e)}")
        raise
except Exception as e:
    logger.error(f"Error in BedrockAgentStack initialization: {str(e)}")
    raise
