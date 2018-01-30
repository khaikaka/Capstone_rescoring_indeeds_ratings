import boto3

def upload_data(file_name):
    s3 = boto3.resource('s3')
    bucket_name = "ha-galvanize-test"
    try:
        s3.meta.client.upload_file(file_name, bucket_name, file_name)

    except Exception as e:
        print(e)  # Note: BucketAlreadyOwnedByYou means you already created the bucket.


if __name__=="__main__":
    s3 = boto3.resource('s3')
    upload_data(FileName)
