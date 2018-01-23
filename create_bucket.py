import boto3
import tinys3

def create_bucket(bucket_name):
    try:
        s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': 'us-west-2'},
    )
    except Exception as e:
        print(e)  # Note: BucketAlreadyOwnedByYou means you already created the bucket.


if __name__=="__main__":
    s3=boto3.client("s3")
    create_bucket('capstone_raw_data')
