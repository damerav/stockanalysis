#!/bin/bash
# Automated AWS EC2 deployment for SPY/ES relay + dashboards
# Usage: ./deploy_aws.sh [region] [api-key]
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed locally

set -e

REGION="${1:-us-east-1}"
API_KEY="${2:-$(openssl rand -hex 16)}"
ADMIN_KEY="$(openssl rand -hex 16)"
APP_NAME="spy-es-relay"
ECR_REPO="$APP_NAME"
INSTANCE_TYPE="t3.micro"
AMI_ID=""  # Will be auto-detected

echo "=== SPY/ES Cloud Deployment ==="
echo "Region: $REGION"
echo "API Key: $API_KEY"
echo "Admin Key: $ADMIN_KEY"

# Step 1: Create ECR repository
echo ""
echo "--- Step 1: ECR Repository ---"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"
echo "ECR URI: $ECR_URI"

# Step 2: Build and push Docker image
echo ""
echo "--- Step 2: Build & Push Docker Image ---"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker build -t "$APP_NAME" -f cloud/Dockerfile .
docker tag "$APP_NAME:latest" "$ECR_URI:latest"
docker push "$ECR_URI:latest"

# Step 3: Create security group
echo ""
echo "--- Step 3: Security Group ---"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region "$REGION")
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$APP_NAME-sg" --query "SecurityGroups[0].GroupId" --output text --region "$REGION" 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group --group-name "$APP_NAME-sg" --description "SPY/ES Relay" --vpc-id "$VPC_ID" --region "$REGION" --query "GroupId" --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$REGION"
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8501 --cidr 0.0.0.0/0 --region "$REGION"
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8502 --cidr 0.0.0.0/0 --region "$REGION"
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION"
fi
echo "Security Group: $SG_ID"

# Step 4: Get latest Amazon Linux 2023 AMI
echo ""
echo "--- Step 4: AMI Selection ---"
AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text --region "$REGION")
echo "AMI: $AMI_ID"

# Step 5: Create IAM role for ECR access
echo ""
echo "--- Step 5: IAM Role ---"
ROLE_NAME="$APP_NAME-ec2-role"
INSTANCE_PROFILE="$APP_NAME-profile"

aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null || \
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }'

aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>/dev/null || true

aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" 2>/dev/null || \
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE"

aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --role-name "$ROLE_NAME" 2>/dev/null || true
sleep 10  # wait for IAM propagation

# Step 6: Launch EC2 instance
echo ""
echo "--- Step 6: Launch EC2 ---"
USER_DATA=$(cat <<EOF
#!/bin/bash
yum install -y docker
systemctl start docker
systemctl enable docker
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker pull $ECR_URI:latest
docker run -d --restart unless-stopped \
    -p 8000:8000 -p 8501:8501 -p 8502:8502 \
    -e RELAY_API_KEY=$API_KEY \
    -e RELAY_ADMIN_KEY=$ADMIN_KEY \
    -e RELAY_URL=http://localhost:8000 \
    $ECR_URI:latest
EOF
)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP_NAME}]" \
    --region "$REGION" \
    --query "Instances[0].InstanceId" --output text)

echo "Instance: $INSTANCE_ID"
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region "$REGION")

echo ""
echo "========================================="
echo "  DEPLOYMENT COMPLETE"
echo "========================================="
echo ""
echo "  Relay URL:     http://$PUBLIC_IP:8000"
echo "  SPY Dashboard: http://$PUBLIC_IP:8501"
echo "  ES Dashboard:  http://$PUBLIC_IP:8502"
echo "  Health Check:  http://$PUBLIC_IP:8000/health"
echo ""
echo "  API Key:       $API_KEY"
echo "  Admin Key:     $ADMIN_KEY"
echo ""
echo "  Add to config.yaml:"
echo "    sync:"
echo "      enabled: true"
echo "      relay_url: \"http://$PUBLIC_IP:8000\""
echo "      api_key: \"$API_KEY\""
echo ""
echo "  Estimated cost: ~\$8-10/month (t3.micro)"
echo "========================================="
