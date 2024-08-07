from nada_dsl import *


def nada_main():
    # Step 1
    user = Party(name="TheUser")  # party 0
    system = Party(name="System")  # party 1

    # Step 2
    ticket_code = SecretInteger(Input(name="ticket_code", party=system))
    timestamp = SecretInteger(Input(name="timestamp", party=system))
    power1 = PublicInteger(Input(name="power1", party=system)) #10
    power2 = PublicInteger(Input(name="power2", party=system)) #8

    # Step 3
    user_otp = SecretInteger(Input(name="otp", party=user))
    ###################################
    #    Hashing                      #
    ###################################
    #denom = 300 * 300
    # def generate_system_otp(ticket_code: SecretInteger,timestamp: SecretInteger) -> SecretInteger:
    # time multiplier denominator
    # denom = 300 * 300
    # timestamp = timestamp / denom
    # multiplier_const = ticket_code*1000000
    # combined = multiplier_const + timestamp_second_trimmer
    
    # # initialize hash value
    # hash_value = 5381

    # for i in range(10):
    #     hash_value = ((hash_value * 33) + hash_value) + (combined % 256)
    #     combined = combined / 256
    #     hash_value = (hash_value * 33) + (combined % 65536)
    #     combined = combined / 65536
    #     hash_value = (hash_value + 2128394518) + (hash_value << 12)
    #     hash_value = (hash_value ^ 3318952252) ^ (hash_value >> 19)
    #     hash_value = (hash_value + 372513201) + (hash_value << 5)
    #     hash_value = (hash_value ^ 3041331308) ^ (hash_value >> 16)

    #     # Ensure the result is positive without using abs()
    # hash_value = (hash_value + (1 << 31)) % (1 << 32)
    # system_otp = hash_value % 1000000
    # return system_otp



    timestamp = timestamp / Integer(300)
    combined = timestamp + ticket_code

    hash_value=Integer(5381)

    for i in range(10):
        hash_value_multiplier = hash_value * Integer(33)
        hash_value = hash_value_multiplier + combined
        power = power1 ** power2
        hash_value = hash_value % power

    system_otp = hash_value % Integer(1000000)



    isvalid=(user_otp==system_otp).if_else(Integer(1),Integer(0))
    out = Output(isvalid, "isvalid", user)
    out2=Output(system_otp, "otp", user)
    return [out,out2]