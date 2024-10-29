# Define variables for clarity and reuse
AWS_NAME=
IMAGE_NAME=
CONTAINER_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# SSH configuration
SSH_USER=
EC2_HOST=
SSH_KEY_PATH=
# A combined target to SSH into the EC2 instance, clean up, prune, pull the latest image, and run the container
.PHONY: deploy
deploy:
	@echo "Connecting to EC2 instance..."
	ssh -i $(SSH_KEY_PATH) $(SSH_USER)@$(EC2_HOST) "\
		echo 'Stopping and removing existing container if it exists...'; \
		sudo docker stop $(CONTAINER_NAME) || true; \
		sudo docker rm $(CONTAINER_NAME) || true; \
		echo 'Pruning unused Docker resources...'; \
		sudo docker system prune -a -f; \
		echo 'Logging into AWS...'; \
		aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin $(AWS_NAME); \
		echo 'Pulling the latest image...'; \
		sudo docker pull $(AWS_NAME)$(IMAGE_NAME); \
		echo 'Running the new container...'; \
		sudo docker run -d -p 8501:8501 -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) --name $(CONTAINER_NAME) $(AWS_NAME)$(IMAGE_NAME); \
		echo 'Container $(CONTAINER_NAME) is up and running.'"
	@echo "Deployment completed successfully."

