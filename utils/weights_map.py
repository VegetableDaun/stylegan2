import numpy as np

available_weights = ['ffhq', 'car', 'cat', 'church', 'horse', 'MNIST']
weights_stylegan2_dir = 'weights/'

mapping_weights = ['Generator/Mapping_network/Conditional_Dense/weight', 'Generator/Mapping_network/Conditional_Dense/bias',
                   'Generator/Mapping_network/Dense0/weight', 'Generator/Mapping_network/Dense0/bias',
                   'Generator/Mapping_network/Dense1/weight', 'Generator/Mapping_network/Dense1/bias',
                   'Generator/Mapping_network/Dense2/weight', 'Generator/Mapping_network/Dense2/bias',
                   'Generator/Mapping_network/Dense3/weight', 'Generator/Mapping_network/Dense3/bias',
                   'Generator/Mapping_network/Dense4/weight', 'Generator/Mapping_network/Dense4/bias',
                   'Generator/Mapping_network/Dense5/weight', 'Generator/Mapping_network/Dense5/bias',
                   'Generator/Mapping_network/Dense6/weight', 'Generator/Mapping_network/Dense6/bias',
                   'Generator/Mapping_network/Dense7/weight', 'Generator/Mapping_network/Dense7/bias']


def get_synthesis_name_weights(resolution):
    synthesis_weights = ['Generator/Synthesis_network/4x4/Const/const',
                         'Generator/Synthesis_network/4x4/Conv1/noise_strength',
                         'Generator/Synthesis_network/4x4/Conv1/bias',
                         'Generator/Synthesis_network/4x4/Conv1/mod_bias',
                         'Generator/Synthesis_network/4x4/Conv1/mod_weight',
                         'Generator/Synthesis_network/4x4/Conv1/weight',
                         'Generator/Synthesis_network/4x4/ToRGB/bias',
                         'Generator/Synthesis_network/4x4/ToRGB/mod_bias',
                         'Generator/Synthesis_network/4x4/ToRGB/mod_weight',
                         'Generator/Synthesis_network/4x4/ToRGB/weight']

    for res in range(3, int(np.log2(resolution)) + 1):
        name = 'Generator/Synthesis_network/{}x{}/'.format(2 ** res, 2 ** res)
        for up in ['Conv0_up/', 'Conv1/', 'ToRGB/']:
            for var in ['noise_strength', 'bias', 'mod_bias', 'mod_weight', 'weight']:
                if up == 'ToRGB/' and var == 'noise_strength':
                    continue
                synthesis_weights.append(name + up + var)

    return synthesis_weights


synthesis_weights_1024 = get_synthesis_name_weights(1024)
synthesis_weights_512 = get_synthesis_name_weights(512)
synthesis_weights_256 = get_synthesis_name_weights(256)
synthesis_weights_32 = get_synthesis_name_weights(32)

discriminator_weights_1024 = ['disc_4x4/Conv/bias',
                              'disc_1024x1024/FromRGB/bias',
                              'disc_1024x1024/FromRGB/weight',
                              'disc_1024x1024/Conv0/bias',
                              'disc_1024x1024/Conv1_down/bias',
                              'disc_1024x1024/Conv0/weight',
                              'disc_1024x1024/Conv1_down/weight',
                              'disc_1024x1024/Skip/weight',
                              'disc_512x512/Conv0/bias',
                              'disc_512x512/Conv1_down/bias',
                              'disc_512x512/Conv0/weight',
                              'disc_512x512/Conv1_down/weight',
                              'disc_512x512/Skip/weight',
                              'disc_256x256/Conv0/bias',
                              'disc_256x256/Conv1_down/bias',
                              'disc_256x256/Conv0/weight',
                              'disc_256x256/Conv1_down/weight',
                              'disc_256x256/Skip/weight',
                              'disc_128x128/Conv0/bias',
                              'disc_128x128/Conv1_down/bias',
                              'disc_128x128/Conv0/weight',
                              'disc_128x128/Conv1_down/weight',
                              'disc_128x128/Skip/weight',
                              'disc_64x64/Conv0/bias',
                              'disc_64x64/Conv1_down/bias',
                              'disc_64x64/Conv0/weight',
                              'disc_64x64/Conv1_down/weight',
                              'disc_64x64/Skip/weight',
                              'disc_32x32/Conv0/bias',
                              'disc_32x32/Conv1_down/bias',
                              'disc_32x32/Conv0/weight',
                              'disc_32x32/Conv1_down/weight',
                              'disc_32x32/Skip/weight',
                              'disc_16x16/Conv0/bias',
                              'disc_16x16/Conv1_down/bias',
                              'disc_16x16/Conv0/weight',
                              'disc_16x16/Conv1_down/weight',
                              'disc_16x16/Skip/weight',
                              'disc_8x8/Conv0/bias',
                              'disc_8x8/Conv1_down/bias',
                              'disc_8x8/Conv0/weight',
                              'disc_8x8/Conv1_down/weight',
                              'disc_8x8/Skip/weight',
                              'disc_4x4/Conv/weight',
                              'disc_4x4/Dense0/weight',
                              'disc_4x4/Dense0/bias',
                              'disc_Output/weight',
                              'disc_Output/bias']

discriminator_weights_512 = ['disc_4x4/Conv/bias',
                             'disc_512x512/FromRGB/bias',
                             'disc_512x512/FromRGB/weight',
                             'disc_512x512/Conv0/bias',
                             'disc_512x512/Conv1_down/bias',
                             'disc_512x512/Conv0/weight',
                             'disc_512x512/Conv1_down/weight',
                             'disc_512x512/Skip/weight',
                             'disc_256x256/Conv0/bias',
                             'disc_256x256/Conv1_down/bias',
                             'disc_256x256/Conv0/weight',
                             'disc_256x256/Conv1_down/weight',
                             'disc_256x256/Skip/weight',
                             'disc_128x128/Conv0/bias',
                             'disc_128x128/Conv1_down/bias',
                             'disc_128x128/Conv0/weight',
                             'disc_128x128/Conv1_down/weight',
                             'disc_128x128/Skip/weight',
                             'disc_64x64/Conv0/bias',
                             'disc_64x64/Conv1_down/bias',
                             'disc_64x64/Conv0/weight',
                             'disc_64x64/Conv1_down/weight',
                             'disc_64x64/Skip/weight',
                             'disc_32x32/Conv0/bias',
                             'disc_32x32/Conv1_down/bias',
                             'disc_32x32/Conv0/weight',
                             'disc_32x32/Conv1_down/weight',
                             'disc_32x32/Skip/weight',
                             'disc_16x16/Conv0/bias',
                             'disc_16x16/Conv1_down/bias',
                             'disc_16x16/Conv0/weight',
                             'disc_16x16/Conv1_down/weight',
                             'disc_16x16/Skip/weight',
                             'disc_8x8/Conv0/bias',
                             'disc_8x8/Conv1_down/bias',
                             'disc_8x8/Conv0/weight',
                             'disc_8x8/Conv1_down/weight',
                             'disc_8x8/Skip/weight',
                             'disc_4x4/Conv/weight',
                             'disc_4x4/Dense0/weight',
                             'disc_4x4/Dense0/bias',
                             'disc_Output/weight',
                             'disc_Output/bias']

discriminator_weights_256 = ['disc_4x4/Conv/bias',
                             'disc_256x256/FromRGB/bias',
                             'disc_256x256/FromRGB/weight',
                             'disc_256x256/Conv0/bias',
                             'disc_256x256/Conv1_down/bias',
                             'disc_256x256/Conv0/weight',
                             'disc_256x256/Conv1_down/weight',
                             'disc_256x256/Skip/weight',
                             'disc_128x128/Conv0/bias',
                             'disc_128x128/Conv1_down/bias',
                             'disc_128x128/Conv0/weight',
                             'disc_128x128/Conv1_down/weight',
                             'disc_128x128/Skip/weight',
                             'disc_64x64/Conv0/bias',
                             'disc_64x64/Conv1_down/bias',
                             'disc_64x64/Conv0/weight',
                             'disc_64x64/Conv1_down/weight',
                             'disc_64x64/Skip/weight',
                             'disc_32x32/Conv0/bias',
                             'disc_32x32/Conv1_down/bias',
                             'disc_32x32/Conv0/weight',
                             'disc_32x32/Conv1_down/weight',
                             'disc_32x32/Skip/weight',
                             'disc_16x16/Conv0/bias',
                             'disc_16x16/Conv1_down/bias',
                             'disc_16x16/Conv0/weight',
                             'disc_16x16/Conv1_down/weight',
                             'disc_16x16/Skip/weight',
                             'disc_8x8/Conv0/bias',
                             'disc_8x8/Conv1_down/bias',
                             'disc_8x8/Conv0/weight',
                             'disc_8x8/Conv1_down/weight',
                             'disc_8x8/Skip/weight',
                             'disc_4x4/Conv/weight',
                             'disc_4x4/Dense0/weight',
                             'disc_4x4/Dense0/bias',
                             'disc_Output/weight',
                             'disc_Output/bias']

discriminator_weights_32 = ['Discriminator/4x4/Conv/bias',
                            'Discriminator/32x32/FromRGB/bias',
                            'Discriminator/32x32/FromRGB/weight',
                            'Discriminator/32x32/Conv0/bias',
                            'Discriminator/32x32/Conv1_down/bias',
                            'Discriminator/32x32/Conv0/weight',
                            'Discriminator/32x32/Conv1_down/weight',
                            'Discriminator/32x32/Skip/weight',
                            'Discriminator/16x16/Conv0/bias',
                            'Discriminator/16x16/Conv1_down/bias',
                            'Discriminator/16x16/Conv0/weight',
                            'Discriminator/16x16/Conv1_down/weight',
                            'Discriminator/16x16/Skip/weight',
                            'Discriminator/8x8/Conv0/bias',
                            'Discriminator/8x8/Conv1_down/bias',
                            'Discriminator/8x8/Conv0/weight',
                            'Discriminator/8x8/Conv1_down/weight',
                            'Discriminator/8x8/Skip/weight',
                            'Discriminator/4x4/Conv/weight',
                            'Discriminator/4x4/Dense0/weight',
                            'Discriminator/4x4/Dense0/bias',
                            'Discriminator/Output_c/weight',
                            'Discriminator/Output_c/bias',
                            'Discriminator/Output_uc/weight',
                            'Discriminator/Output_uc/bias']

synthesis_weights = {
    'ffhq': synthesis_weights_1024,
    'car': synthesis_weights_512,
    'cat': synthesis_weights_256,
    'horse': synthesis_weights_256,
    'church': synthesis_weights_256,
    'MNIST' : synthesis_weights_32,
    32 : synthesis_weights_32
}

discriminator_weights = {
    'ffhq': discriminator_weights_1024,
    'car': discriminator_weights_512,
    'cat': discriminator_weights_256,
    'horse': discriminator_weights_256,
    'church': discriminator_weights_256,
    'MNIST': discriminator_weights_32,
    32: discriminator_weights_32}
