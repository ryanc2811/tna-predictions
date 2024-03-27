
export GCLOUD_PROJECT="glptraining" 

export REPO="glp-tna"

export REGION="europe-west2"

export IMAGE="tna-project-image"


export IMAGE_TAG=${REGION}-docker.pkg.dev/$GCLOUD_PROJECT/$REPO

docker build -t $IMAGE_TAG --platform linux/x86_64 .
# Push it to Artifact Registry:
docker push $IMAGE_TAG