import {
    Stack,
    StackProps,
    aws_sqs,
    CfnOutput,
    aws_lambda,
    aws_lambda_event_sources
} from "aws-cdk-lib";
import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as pythonLambda from "@aws-cdk/aws-lambda-python-alpha";

export class DataPrepStack extends Stack {
    constructor(scope: Construct, id: string, props?: StackProps) {
        super(scope, id, props);

        const dlq = new aws_sqs.Queue(this, "deadLetterQueue", {});
        new CfnOutput(this, "deadLetterQueueUrl", { value: dlq.queueUrl });

        const dataPrepQueue = new aws_sqs.Queue(this, "DataPrepQueue", {
            visibilityTimeout: cdk.Duration.minutes(30),
            deadLetterQueue: { queue: dlq, maxReceiveCount: 2 }
        });

        // See https://docs.aws.amazon.com/cdk/api/v2/docs/aws-lambda-python-alpha-readme.html
        const dataPrepLambda = new pythonLambda.PythonFunction(this, "DataPrepLambda", {
            entry: "/path/to/my/function", // required
            runtime: aws_lambda.Runtime.PYTHON_3_8, // required
            index: "my_index.py", // optional, defaults to 'index.py'
            handler: "my_exported_func" // optional, defaults to 'handler'
        });

        dataPrepQueue.grantConsumeMessages(dataPrepLambda);
        dataPrepLambda.addEventSource(
            new aws_lambda_event_sources.SqsEventSource(dataPrepQueue, { batchSize: 1 })
        );
        new CfnOutput(this, "DataPrepQueueUrl", { value: dataPrepQueue.queueUrl });
    }
}
