# the-logistics-prophet Terraform

Minimal Cloud Run deployment skeleton for `the-logistics-prophet`.

## Apply

```bash
terraform init
terraform apply \
  -var="project_id=your-project" \
  -var="image=asia-northeast3-docker.pkg.dev/your-project/apps/the-logistics-prophet:latest"
```

Use `env` to inject Streamlit secrets, Ollama/Datadog options, and service-store paths.
