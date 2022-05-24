#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { DataPrepStack } from "../lib/data-prep-stack";

const app = new cdk.App();
new DataPrepStack(app, "DataPrepStack", {
    /* Uncomment the next line to specialize this stack for the AWS Account
     * and Region that are implied by the current CLI configuration. */
    env: {
        account: process.env.CDK_DEFAULT_ACCOUNT,
        region: "us-west-2"
    }
});
