PY        ?= python
STACK     ?= defect-dashboard
# us-east-2: the AWS organization's SCP blocks CloudFormation in us-east-1.
REGION    ?= us-east-2
# Set CREATE_OIDC=false when the GitHub OIDC provider already exists (e.g. created by an org admin).
CREATE_OIDC ?= true
DATA_URI  ?= data
SAM       ?= sam
BUCKET     = $(shell aws cloudformation describe-stacks --stack-name $(STACK) --region $(REGION) \
               --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text 2>/dev/null)

.PHONY: install lint test package validate build deploy bootstrap teardown ingest backfill train app bench

install:            ## dev environment
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

test:
	pytest --cov --cov-report=term-missing

# ---------------------------------------------------------------- AWS
package:            ## assemble the Lambda source tree (only what the function imports)
	rm -rf build/lambda && mkdir -p build/lambda
	cp -R common ingestion transform build/lambda/
	# Vendor deps as Linux/arm64 wheels (no Docker needed; pandas+pyarrow come from the AWS layer)
	$(PY) -m pip install -q -r ingestion/requirements-lambda.txt --target build/lambda \
	  --platform manylinux2014_aarch64 --implementation cp --python-version 3.12 --only-binary=:all:
	find build/lambda -name '__pycache__' -prune -exec rm -rf {} +

validate:
	cd infra && $(SAM) validate --lint --region $(REGION)
	cd infra && $(SAM) validate --lint --region $(REGION) --template bootstrap.yaml

build: package
	cd infra && $(SAM) build

deploy: build       ## make deploy ALERT_EMAIL=you@example.com
	cd infra && $(SAM) deploy --stack-name $(STACK) --region $(REGION) \
	  --parameter-overrides AlertEmail=$(ALERT_EMAIL)

bootstrap:          ## one-time: GitHub OIDC roles.  make bootstrap GITHUB_REPO=owner/repo [CREATE_OIDC=false]
	aws cloudformation deploy --template-file infra/bootstrap.yaml --stack-name $(STACK)-bootstrap \
	  --region $(REGION) --capabilities CAPABILITY_NAMED_IAM \
	  --parameter-overrides GitHubRepo=$(GITHUB_REPO) CreateOIDCProvider=$(CREATE_OIDC)

teardown:           ## delete EVERYTHING (data included). Irreversible.
	@test -n "$(BUCKET)" || (echo "stack $(STACK) not found in $(REGION)"; exit 1)
	aws s3 rm s3://$(BUCKET) --recursive --region $(REGION)
	$(SAM) delete --stack-name $(STACK) --region $(REGION) --no-prompts
	-aws cloudformation delete-stack --stack-name $(STACK)-bootstrap --region $(REGION)
	-aws ssm delete-parameter --name /defect-dashboard/github-token --region $(REGION)

# ---------------------------------------------------------------- jobs (local or against s3://)
ingest:
	$(PY) -m ingestion.run --store $(DATA_URI)

backfill:
	$(PY) -m ingestion.run --store $(DATA_URI) --until-caught-up --max-pages 100

train:
	$(PY) -m ml.train --store $(DATA_URI)

app:
	DATA_URI=$(DATA_URI) streamlit run app/streamlit_app.py

bench:
	$(PY) -m benchmarks.cold_load
