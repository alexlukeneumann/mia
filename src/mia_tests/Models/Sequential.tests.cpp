#include <CppUnitTest.h>

#include <Models/Sequential.h>
#include <Layers/Layer.h>
#include <Layers/InputLayer.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        static u32 constexpr c_TestSeedValue = 0;

        class DummyBaseLayer
        {
        public:
            u32 m_CompileCalls = 0;
            u32 m_ExecuteCalls = 0;

            static u32 m_DestructorCalls;
        };

        u32 DummyBaseLayer::m_DestructorCalls = 0;

        class DummyInputLayer : public layers::InputLayer, public DummyBaseLayer
        {
        public:
            DummyInputLayer()
                : InputLayer(activators::ActivatorType::None)
            {
            }

            virtual ~DummyInputLayer()
            {
                m_DestructorCalls++;
            }

            virtual void Compile(u32 seedValue, layers::Layer const * prevLayer) override
            {
                Assert::IsNull(prevLayer);
                m_CompileCalls++;
            }

            virtual void Execute(layers::Layer const * prevLayer) override
            {
                Assert::IsNull(prevLayer);
                m_ExecuteCalls++;
            }

            virtual void SetInputData(NDArrayView<f32> const& inputData) override
            {
                m_SetInputDataCalls++;
            }

            u32 m_SetInputDataCalls = 0;
        };

        class DummyLayer : public layers::Layer, public DummyBaseLayer
        {
        public:
            DummyLayer()
                : Layer(activators::ActivatorType::None)
            {
            }

            virtual ~DummyLayer()
            {
                m_DestructorCalls++;
            }

            virtual void Compile(u32 seedValue, layers::Layer const * prevLayer) override
            {
                DummyBaseLayer const * prev = dynamic_cast<DummyBaseLayer const *>(prevLayer);
                if (nullptr != prev)
                {
                    Assert::AreEqual(static_cast<u32>(1), prev->m_CompileCalls);
                    m_CompileCalls++;
                    return;
                }

                Assert::IsTrue(false, L"Unknown previous layer in test.");
            }

            virtual void Execute(layers::Layer const * prevLayer) override
            {
                DummyBaseLayer const * prev = dynamic_cast<DummyBaseLayer const *>(prevLayer);
                if (nullptr != prev)
                {
                    Assert::AreEqual(static_cast<u32>(1), prev->m_ExecuteCalls);
                    m_ExecuteCalls++;
                    return;
                }

                Assert::IsTrue(false, L"Unknown previous layer in test.");
            }
        };

        TEST_CLASS(SequentialTests)
        {
            static u32 constexpr c_TestSeedValue = 0;

        public:
            TEST_METHOD(Compile_CallsCompileSequentiallyThroughTheLayers)
            {
                DummyInputLayer * layer0 =  new DummyInputLayer();
                DummyLayer * layer1 = new DummyLayer();
                DummyLayer * layer2 = new DummyLayer();

                models::Sequential model({
                    layer0,
                    layer1,
                    layer2
                });

                model.Compile(c_TestSeedValue);

                Assert::AreEqual(static_cast<u32>(1), layer0->m_CompileCalls);
                Assert::AreEqual(static_cast<u32>(1), layer1->m_CompileCalls);
                Assert::AreEqual(static_cast<u32>(1), layer2->m_CompileCalls);
            }

            TEST_METHOD(OnDestruction_DestructorDeletesAllLayers)
            {
                DummyInputLayer * layer0 =  new DummyInputLayer();
                DummyLayer * layer1 = new DummyLayer();
                DummyLayer * layer2 = new DummyLayer();

                // Set destructor call values to zero
                DummyBaseLayer::m_DestructorCalls = 0;

                Assert::AreEqual(static_cast<u32>(0), DummyBaseLayer::m_DestructorCalls);

                // Artificially scope the model to cause a destruction
                {
                    models::Sequential model({
                        layer0,
                        layer1,
                        layer2
                    });
                }

                Assert::AreEqual(static_cast<u32>(3), DummyBaseLayer::m_DestructorCalls);

                // Reset values back to zero
                DummyInputLayer::m_DestructorCalls = 0;
                DummyLayer::m_DestructorCalls = 0;
            }

            TEST_METHOD(Train_CallsSetInputDataOnFirstLayer)
            {
                DummyInputLayer * layer0 =  new DummyInputLayer();
                DummyLayer * layer1 = new DummyLayer();
                DummyLayer * layer2 = new DummyLayer();

                models::Sequential model({
                    layer0,
                    layer1,
                    layer2
                    });

                model.Compile(c_TestSeedValue);

                Assert::AreEqual(static_cast<u32>(0), layer0->m_SetInputDataCalls);
                model.Train({}, {});
                Assert::AreEqual(static_cast<u32>(1), layer0->m_SetInputDataCalls);
            }

            TEST_METHOD(Train_CallsExecuteOnEachLayerSequentially)
            {
                DummyInputLayer * layer0 =  new DummyInputLayer();
                DummyLayer * layer1 = new DummyLayer();
                DummyLayer * layer2 = new DummyLayer();

                models::Sequential model({
                    layer0,
                    layer1,
                    layer2
                    });

                model.Compile(c_TestSeedValue);
                model.Train({}, {});

                Assert::AreEqual(static_cast<u32>(1), layer0->m_ExecuteCalls);
                Assert::AreEqual(static_cast<u32>(1), layer1->m_ExecuteCalls);
                Assert::AreEqual(static_cast<u32>(1), layer2->m_ExecuteCalls);
            }
        };
    }
}
