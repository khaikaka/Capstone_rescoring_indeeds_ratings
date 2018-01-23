import boto3

def upload_data(file_name):
    s3 = boto3.resource('s3')
    bucket_name = "capstone_raw_data"
    try:
        s3.meta.client.upload_file(file_name, bucket_name, file_name)

    except Exception as e:
        print(e)  # Note: BucketAlreadyOwnedByYou means you already created the bucket.
